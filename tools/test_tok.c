#include <stdio.h>
#include <stdlib.h>
#include "tokenizer.h"

int main() {
    Tokenizer t;
    tok_load(&t, "/Users/deepanwadhwa/.samosa/current/model/tokenizer.json");
    printf("image_pad ID: %d\n", tok_id_of(&t, "<|image_pad|>"));
    return 0;
}
