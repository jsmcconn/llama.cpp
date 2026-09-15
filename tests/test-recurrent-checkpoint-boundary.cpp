#include "common.h"
#include <cstdlib>
#include <cstdio>

int main() {
    common_prompt_checkpoint c;
    c.update_pos(4096, 0, 4095);
    auto check = [](bool condition, const char * description) {
        if (!condition) { fprintf(stderr, "%s\n", description); std::exit(1); }
    };
    check(c.can_resume_recurrent(4096, 4100), "append must reuse the entire saved prefix");
    check(c.can_resume_recurrent(6000, 7000), "older exact checkpoint must survive a later edit");
    check(!c.can_resume_recurrent(2000, 7000), "edit before checkpoint cannot rewind recurrent state");
    check(!c.can_resume_recurrent(4096, 4096), "identical retry requires logits from a real suffix");
    check(!c.can_resume_recurrent(2048, 2048), "compacted prefix cannot use a later final snapshot");
    c.update_pos(4096, 4095, 4095);
    check(c.can_resume_recurrent(4096, 5000), "mid-prompt recurrent boundary uses inclusive end");
    c.update_pos(4096, 0, 4096);
    check(!c.can_resume_recurrent(4096, 5000), "one-past-end metadata must be rejected");
    c.update_pos(0, 0, -1);
    check(!c.can_resume_recurrent(0, 5000), "empty checkpoint cannot resume");
}
