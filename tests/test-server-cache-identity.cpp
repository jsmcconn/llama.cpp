#include "server-task.h"
#include <cstdio>
#include <cstdlib>

static void check(bool condition, const char * description) {
    if (!condition) { fprintf(stderr, "%s\n", description); std::exit(1); }
}
int main() {
    server_prompt_cache cache(16, 1024);
    server_prompt a;
    a.tokens = server_tokens(llama_tokens{1, 2, 3, 4}, false);
    check(cache.alloc(a, 0, 0, "alice") != nullptr, "first owner must allocate");
    check(cache.alloc(a, 0, 0, "bob") != nullptr, "identical tokens from a second owner must remain separate");
    check(cache.alloc(a, 0, 0, "alice") == nullptr, "deduplication must still work for the same owner");
    a.tokens.push_back(5);
    check(cache.alloc(a, 0, 0, "alice") != nullptr, "owner may extend its saved prompt");
    check(cache.states.size() == 2, "extension must not evict another owner's prefix as obsolete");
    check(cache.states.front().user_id == "bob", "other owner's entry must survive");
    server_prompt empty;
    cache.load(empty, a.tokens, nullptr, nullptr, 0, "charlie");
    check(empty.tokens.empty() && cache.states.size() == 2, "third owner cannot restore another owner's state");
    // A media-capable empty/text container must not assert or silently lose tokens.
    server_tokens text(llama_tokens{1, 2, 3}, true);
    check(text.get_tokens().size() == 3, "text access is based on content, not media capability");
}
