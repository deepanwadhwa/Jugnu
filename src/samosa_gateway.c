/* Compiled Samosa gateway: local app, backend supervision, and raw API proxy. */
#define _GNU_SOURCE
#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "json.h"
#include "samosa_http.h"

typedef struct {
    SamosaHttpServer *server;
    pthread_mutex_t mu;
    pid_t backend_pid;
    int upstream_fd;
    atomic_int generating;
    atomic_int stopping;
    int public_port;
    int backend_port;
    char home[PATH_MAX];
    char backend[16];
    char app_html[PATH_MAX];
    char app_logo[PATH_MAX];
    char qwen_engine[PATH_MAX];
    char qwen_model[PATH_MAX];
    char tokenizer[PATH_MAX];
    char llama_server[PATH_MAX];
    char bonsai_model[PATH_MAX];
    char ornith_model[PATH_MAX];
    char backend_log[PATH_MAX];
    char selection_file[PATH_MAX];
} Gateway;

static Gateway *signal_gateway;

static int path_copy(char *out, size_t cap, const char *value) {
    int n = snprintf(out, cap, "%s", value ? value : "");
    return n >= 0 && (size_t)n < cap;
}

static int path_join(char *out, size_t cap, const char *left, const char *right) {
    int n = snprintf(out, cap, "%s/%s", left, right);
    return n >= 0 && (size_t)n < cap;
}

static int regular_file(const char *path, int executable) {
    struct stat st;
    return path && !stat(path, &st) && S_ISREG(st.st_mode) &&
           (!executable || !access(path, X_OK));
}

static int mkdirs(const char *path) {
    char copy[PATH_MAX];
    if (!path_copy(copy, sizeof(copy), path)) return 0;
    for (char *p = copy + 1; *p; ++p) {
        if (*p != '/') continue;
        *p = 0;
        if (mkdir(copy, 0700) && errno != EEXIST) return 0;
        *p = '/';
    }
    return !mkdir(copy, 0700) || errno == EEXIST;
}

static int read_small_file(const char *path, char *out, size_t cap) {
    int fd = open(path, O_RDONLY | O_NOFOLLOW);
    if (fd < 0) return 0;
    ssize_t n = read(fd, out, cap - 1);
    close(fd);
    if (n < 0) return 0;
    out[n] = 0;
    while (n > 0 && (out[n - 1] == '\n' || out[n - 1] == '\r' || out[n - 1] == ' '))
        out[--n] = 0;
    return n > 0;
}

static int write_small_file(const char *path, const char *text) {
    char temp[PATH_MAX];
    if (snprintf(temp, sizeof(temp), "%s.tmp.%ld", path, (long)getpid()) >=
        (int)sizeof(temp)) return 0;
    int out = open(temp, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
    if (out < 0) return 0;
    size_t length = strlen(text), written = 0;
    int ok = 1;
    while (written < length) {
        ssize_t n = write(out, text + written, length - written);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { ok = 0; break; }
        written += (size_t)n;
    }
    if (ok) ok = fsync(out) == 0;
    if (close(out)) ok = 0;
    if (ok) ok = rename(temp, path) == 0;
    if (!ok) unlink(temp);
    return ok;
}

static int backend_available(Gateway *g, const char *name) {
    if (!strcmp(name, "qwen")) {
        char experts[PATH_MAX];
        return path_join(experts, sizeof(experts), g->qwen_model, "experts.bin") &&
               regular_file(g->qwen_engine, 1) && regular_file(experts, 0);
    }
    if (!strcmp(name, "bonsai"))
        return regular_file(g->llama_server, 1) && regular_file(g->bonsai_model, 0);
    if (!strcmp(name, "ornith"))
        return regular_file(g->llama_server, 1) && regular_file(g->ornith_model, 0);
    return 0;
}

static int tcp_connect(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_in address = {0};
    address.sin_family = AF_INET;
    address.sin_port = htons((uint16_t)port);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(fd, (struct sockaddr *)&address, sizeof(address))) {
        close(fd);
        return -1;
    }
    return fd;
}

static int backend_probe(Gateway *g) {
    int fd = tcp_connect(g->backend_port);
    if (fd < 0) return 0;
    const char *path = !strcmp(g->backend, "qwen") ? "/healthz" : "/health";
    char request[256];
    int n = snprintf(request, sizeof(request),
                     "GET %s HTTP/1.1\r\nHost: 127.0.0.1:%d\r\nConnection: close\r\n\r\n",
                     path, g->backend_port);
    char response[64] = {0};
    int ok = n > 0 && samosa_send_all(fd, request, (size_t)n) &&
             recv(fd, response, sizeof(response) - 1, 0) > 0 &&
             strstr(response, " 200 ") != NULL;
    close(fd);
    return ok;
}

static void backend_stop(Gateway *g) {
    pthread_mutex_lock(&g->mu);
    pid_t pid = g->backend_pid;
    int upstream = g->upstream_fd;
    g->backend_pid = 0;
    g->upstream_fd = -1;
    pthread_mutex_unlock(&g->mu);
    if (upstream >= 0) shutdown(upstream, SHUT_RDWR);
    if (pid <= 0) return;
    kill(pid, SIGTERM);
    for (int i = 0; i < 80; ++i) {
        if (waitpid(pid, NULL, WNOHANG) == pid) return;
        struct timespec pause = {.tv_sec = 0, .tv_nsec = 100000000};
        nanosleep(&pause, NULL);
    }
    kill(pid, SIGKILL);
    waitpid(pid, NULL, 0);
}

static int backend_start(Gateway *g) {
    if (!backend_available(g, g->backend)) return 0;
    char chats[PATH_MAX];
    if (!path_join(chats, sizeof(chats), g->home, "chats") || !mkdirs(chats)) return 0;
    pid_t pid = fork();
    if (pid < 0) return 0;
    if (pid == 0) {
        int log = open(g->backend_log, O_WRONLY | O_CREAT | O_APPEND, 0600);
        if (log >= 0) { dup2(log, STDOUT_FILENO); dup2(log, STDERR_FILENO); close(log); }
        char port[16];
        snprintf(port, sizeof(port), "%d", g->backend_port);
        if (!strcmp(g->backend, "qwen")) {
            setenv("SNAP", g->qwen_model, 1);
            setenv("TOKENIZER", g->tokenizer, 1);
            setenv("SAMOSA_CHATS_DIR", chats, 1);
            execl(g->qwen_engine, g->qwen_engine, "--serve", "--port", port,
                  "--tokenizer", g->tokenizer, (char *)NULL);
        } else {
            const char *model = !strcmp(g->backend, "ornith") ?
                                g->ornith_model : g->bonsai_model;
            const char *alias = !strcmp(g->backend, "ornith") ?
                                "ornith-1.0-9b" : "bonsai-27b-1bit";
            execl(g->llama_server, g->llama_server, "-m", model, "-ngl", "99",
                  "-c", "8192", "-np", "1", "--cache-ram", "0", "--host",
                  "127.0.0.1", "--port", port, "--no-ui", "--alias", alias,
                  (char *)NULL);
        }
        _Exit(127);
    }
    pthread_mutex_lock(&g->mu);
    g->backend_pid = pid;
    pthread_mutex_unlock(&g->mu);
    return 1;
}

static int static_file(int fd, const char *path, const char *type, const char *extra) {
    int file = open(path, O_RDONLY | O_NOFOLLOW);
    if (file < 0) return 0;
    struct stat st;
    if (fstat(file, &st) || !S_ISREG(st.st_mode) || st.st_size < 0 || st.st_size > (4 << 20)) {
        close(file); return 0;
    }
    size_t size = (size_t)st.st_size;
    char *data = malloc(size ? size : 1);
    if (!data) { close(file); return 0; }
    size_t used = 0;
    while (used < size) {
        ssize_t n = read(file, data + used, size - used);
        if (n <= 0) { free(data); close(file); return 0; }
        used += (size_t)n;
    }
    close(file);
    int ok = samosa_http_headers(fd, 200, type, size, extra) &&
             (!size || samosa_send_all(fd, data, size));
    free(data);
    return ok;
}

static int proxy_request(Gateway *g, int client, const SamosaHttpRequest *request) {
    if (!backend_probe(g))
        return samosa_http_json_error(client, 503, "backend_loading", "The model is still loading.");
    int upstream = tcp_connect(g->backend_port);
    if (upstream < 0)
        return samosa_http_json_error(client, 503, "backend_unavailable", "The model backend is unavailable.");
    pthread_mutex_lock(&g->mu); g->upstream_fd = upstream; pthread_mutex_unlock(&g->mu);
    atomic_store(&g->generating, 1);
    char header[1024];
    int n = snprintf(header, sizeof(header),
        "%s %s HTTP/1.1\r\nHost: 127.0.0.1:%d\r\nContent-Type: application/json\r\n"
        "Content-Length: %zu\r\nConnection: close\r\n\r\n",
        request->method, request->path, g->backend_port, request->body_len);
    int ok = n > 0 && (size_t)n < sizeof(header) &&
             samosa_send_all(upstream, header, (size_t)n) &&
             (!request->body_len || samosa_send_all(upstream, request->body, request->body_len));
    char buffer[65536];
    while (ok) {
        ssize_t got = recv(upstream, buffer, sizeof(buffer), 0);
        if (got == 0) break;
        if (got < 0) { if (errno == EINTR) continue; ok = 0; break; }
        if (!samosa_send_all(client, buffer, (size_t)got)) { ok = 0; break; }
    }
    pthread_mutex_lock(&g->mu);
    if (g->upstream_fd == upstream) g->upstream_fd = -1;
    pthread_mutex_unlock(&g->mu);
    close(upstream);
    atomic_store(&g->generating, 0);
    return ok;
}

static const char *backend_label(const char *name) {
    if (!strcmp(name, "ornith")) return "Ornith 9B";
    if (!strcmp(name, "bonsai")) return "Bonsai 27B 1-bit";
    return "Qwen3.6 35B A3B";
}

static const char *backend_model(const char *name) {
    if (!strcmp(name, "ornith")) return "ornith-1.0-9b";
    if (!strcmp(name, "bonsai")) return "bonsai-27b-1bit";
    return "qwen3.6-35b-a3b";
}

static int gateway_handler(SamosaHttpServer *server, int fd,
                           const SamosaHttpRequest *request, void *opaque) {
    Gateway *g = opaque;
    if (!strcmp(request->method, "GET") &&
        (!strcmp(request->path, "/") || !strcmp(request->path, "/index.html"))) {
        const char *policy = "Content-Security-Policy: default-src 'self'; img-src 'self' data: blob:; "
            "style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'\r\n";
        if (static_file(fd, g->app_html, "text/html; charset=utf-8", policy)) return 1;
        return samosa_http_json_error(fd, 404, "app_missing", "The app asset is missing.");
    }
    if (!strcmp(request->method, "GET") && !strcmp(request->path, "/assets/samosa-chat.png")) {
        if (static_file(fd, g->app_logo, "image/png", NULL)) return 1;
        return samosa_http_json_error(fd, 404, "logo_missing", "The app logo is missing.");
    }
    if (!strcmp(request->method, "GET") && !strcmp(request->path, "/healthz")) {
        char body[768];
        pthread_mutex_lock(&g->mu); pid_t pid = g->backend_pid; pthread_mutex_unlock(&g->mu);
        int ready = backend_probe(g);
        snprintf(body, sizeof(body),
            "{\"gateway\":true,\"compiled\":true,\"backend\":\"%s\","
            "\"label\":\"%s\",\"model\":\"%s\",\"supports_images\":%s,"
            "\"ready\":%s,\"loading\":%s,\"generating\":%s,\"pid\":%ld}",
            g->backend, backend_label(g->backend), backend_model(g->backend),
            !strcmp(g->backend, "qwen") ? "true" : "false",
            ready ? "true" : "false", (!ready && pid > 0) ? "true" : "false",
            atomic_load(&g->generating) ? "true" : "false", (long)pid);
        return samosa_http_response(fd, 200, "application/json", body, NULL);
    }
    if (!strcmp(request->method, "GET") && !strcmp(request->path, "/v1/backends")) {
        char body[1536];
        snprintf(body, sizeof(body),
            "{\"active\":\"%s\",\"backends\":["
            "{\"id\":\"bonsai\",\"label\":\"Bonsai 27B 1-bit\",\"model\":\"bonsai-27b-1bit\",\"supports_images\":false,\"available\":%s},"
            "{\"id\":\"ornith\",\"label\":\"Ornith 9B\",\"model\":\"ornith-1.0-9b\",\"supports_images\":false,\"available\":%s},"
            "{\"id\":\"qwen\",\"label\":\"Qwen3.6 35B A3B\",\"model\":\"qwen3.6-35b-a3b\",\"supports_images\":true,\"available\":%s}]}",
            g->backend, backend_available(g, "bonsai") ? "true" : "false",
            backend_available(g, "ornith") ? "true" : "false",
            backend_available(g, "qwen") ? "true" : "false");
        return samosa_http_response(fd, 200, "application/json", body, NULL);
    }
    if (!strcmp(request->method, "POST") && !strcmp(request->path, "/v1/backends/select")) {
        char *arena = NULL;
        jval *root = json_parse(request->body, &arena);
        jval *selected = root && root->t == J_OBJ ? json_get(root, "backend") : NULL;
        if (!selected || selected->t != J_STR ||
            (strcmp(selected->str, "qwen") && strcmp(selected->str, "bonsai") &&
             strcmp(selected->str, "ornith"))) {
            json_free(root); free(arena);
            return samosa_http_json_error(fd, 400, "invalid_backend", "Unknown model backend.");
        }
        if (!backend_available(g, selected->str)) {
            json_free(root); free(arena);
            return samosa_http_json_error(fd, 409, "backend_unavailable", "That model backend is not installed.");
        }
        char name[16]; path_copy(name, sizeof(name), selected->str);
        json_free(root); free(arena);
        if (atomic_load(&g->generating))
            return samosa_http_json_error(fd, 409, "generation_active", "Stop the current response before switching models.");
        if (strcmp(name, g->backend)) {
            backend_stop(g);
            path_copy(g->backend, sizeof(g->backend), name);
            char persisted[32]; snprintf(persisted, sizeof(persisted), "%s\n", name);
            if (!write_small_file(g->selection_file, persisted) || !backend_start(g))
                return samosa_http_json_error(fd, 500, "backend_start_failed", "The selected model could not be started.");
        }
        return samosa_http_response(fd, 202, "application/json", "{\"accepted\":true}", NULL);
    }
    if (!strcmp(request->method, "POST") && !strcmp(request->path, "/v1/cancel")) {
        pthread_mutex_lock(&g->mu); int upstream = g->upstream_fd; pthread_mutex_unlock(&g->mu);
        if (upstream >= 0) shutdown(upstream, SHUT_RDWR);
        return samosa_http_response(fd, 200, "application/json",
                                    upstream >= 0 ? "{\"cancelled\":true}" : "{\"cancelled\":false}", NULL);
    }
    if (!strcmp(request->method, "POST") &&
        (!strcmp(request->path, "/v1/shutdown") || !strcmp(request->path, "/v1/kill"))) {
        atomic_store(&g->stopping, 1);
        samosa_http_response(fd, 200, "application/json", "{\"stopping\":true}", NULL);
        backend_stop(g);
        samosa_http_server_stop(server);
        return 1;
    }
    if (!strcmp(request->path, "/v1/chat/completions") ||
        !strcmp(request->path, "/v1/models"))
        return proxy_request(g, fd, request);
    if (!strncmp(request->path, "/v1/jobs/", 9))
        return samosa_http_json_error(fd, 503, "jobs_port_in_progress",
                                      "The compiled Jobs controller is not available yet.");
    return samosa_http_json_error(fd, 404, "not_found", "Endpoint not found.");
}

static void on_signal(int number) {
    (void)number;
    if (!signal_gateway) return;
    atomic_store(&signal_gateway->stopping, 1);
    if (signal_gateway->server) samosa_http_server_stop(signal_gateway->server);
}

static int load_config(Gateway *g) {
    memset(g, 0, sizeof(*g));
    g->backend_pid = 0; g->upstream_fd = -1;
    pthread_mutex_init(&g->mu, NULL);
    atomic_init(&g->generating, 0); atomic_init(&g->stopping, 0);
    const char *home = getenv("SAMOSA_HOME");
    const char *user_home = getenv("HOME");
    if (!home) {
        if (!user_home || snprintf(g->home, sizeof(g->home), "%s/.samosa", user_home) >=
                          (int)sizeof(g->home)) return 0;
    } else if (!path_copy(g->home, sizeof(g->home), home)) return 0;
    g->public_port = getenv("SAMOSA_PORT") ? atoi(getenv("SAMOSA_PORT")) : 8642;
    g->backend_port = getenv("SAMOSA_BACKEND_PORT") ? atoi(getenv("SAMOSA_BACKEND_PORT")) : g->public_port + 1;
#define ENV_PATH(field, name, fallback) do { const char *v = getenv(name); \
    if (v) { if (!path_copy(g->field, sizeof(g->field), v)) return 0; } \
    else { if (!path_join(g->field, sizeof(g->field), g->home, fallback)) return 0; } } while (0)
    ENV_PATH(app_html, "SAMOSA_APP_HTML", "current/app.html");
    ENV_PATH(app_logo, "SAMOSA_APP_LOGO", "current/samosa-chat.png");
    ENV_PATH(qwen_engine, "SAMOSA_QWEN_ENGINE", "current/bin/qwen36b");
    ENV_PATH(qwen_model, "SAMOSA_QWEN_MODEL", "current/model");
    ENV_PATH(tokenizer, "SAMOSA_TOKENIZER", "current/tokenizer_qwen36.json");
    ENV_PATH(llama_server, "SAMOSA_BONSAI_SERVER", "backends/prism-llama.cpp/build/bin/llama-server");
    ENV_PATH(bonsai_model, "SAMOSA_BONSAI_MODEL", "models/bonsai-27b-1bit/Bonsai-27B-Q1_0.gguf");
    ENV_PATH(ornith_model, "SAMOSA_ORNITH_MODEL", "models/ornith-9b/Ornith-1.0-9B-Q4_K_M.gguf");
#undef ENV_PATH
    if (!path_join(g->backend_log, sizeof(g->backend_log), g->home, "backend.log") ||
        !path_join(g->selection_file, sizeof(g->selection_file), g->home, "model-backend") ||
        !mkdirs(g->home)) return 0;
    char selected[32] = {0};
    if (read_small_file(g->selection_file, selected, sizeof(selected)) &&
        backend_available(g, selected)) path_copy(g->backend, sizeof(g->backend), selected);
    else if (backend_available(g, "ornith")) path_copy(g->backend, sizeof(g->backend), "ornith");
    else if (backend_available(g, "bonsai")) path_copy(g->backend, sizeof(g->backend), "bonsai");
    else path_copy(g->backend, sizeof(g->backend), "qwen");
    return g->public_port > 0 && g->public_port < 65536 &&
           g->backend_port > 0 && g->backend_port < 65536;
}

int main(void) {
    Gateway gateway;
    if (!load_config(&gateway)) {
        fprintf(stderr, "samosa-gateway: invalid configuration\n"); return 2;
    }
    if (!backend_start(&gateway)) {
        fprintf(stderr, "samosa-gateway: backend %s is not installed\n", gateway.backend); return 2;
    }
    SamosaHttpServer server;
    if (!samosa_http_server_init(&server, gateway.public_port, gateway_handler, &gateway)) {
        fprintf(stderr, "samosa-gateway: cannot bind 127.0.0.1:%d: %s\n",
                gateway.public_port, strerror(errno)); backend_stop(&gateway); return 2;
    }
    gateway.server = &server; signal_gateway = &gateway;
    signal(SIGINT, on_signal); signal(SIGTERM, on_signal);
    fprintf(stderr, "[gateway] compiled ready http://127.0.0.1:%d backend=%s\n",
            server.port, gateway.backend); fflush(stderr);
    int ok = samosa_http_server_run(&server);
    backend_stop(&gateway);
    samosa_http_server_destroy(&server);
    pthread_mutex_destroy(&gateway.mu);
    signal_gateway = NULL;
    return ok ? 0 : 2;
}
