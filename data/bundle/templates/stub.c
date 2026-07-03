/* asiai bundle exec stub — Contents/MacOS/<AppName>.
 *
 * Background Task Management attributes a daemon to the bundle whose
 * executable it runs, so the embedded plists must point INSIDE the bundle.
 * This stub immediately exec()s the real launcher (asiai-launch <service>),
 * whose absolute path is baked in at build time (-DASIAI_LAUNCHER_PATH).
 * Kept in C so the bundle needs no runtime beyond libSystem.
 */
#include <stdio.h>
#include <unistd.h>

#ifndef ASIAI_LAUNCHER_PATH
#error "ASIAI_LAUNCHER_PATH must be defined at compile time"
#endif

int main(int argc, char *argv[]) {
    char *args[argc + 1];
    args[0] = (char *)ASIAI_LAUNCHER_PATH;
    for (int i = 1; i < argc; i++) {
        args[i] = argv[i];
    }
    args[argc] = NULL;
    execv(ASIAI_LAUNCHER_PATH, args);
    perror("asiai stub: execv " ASIAI_LAUNCHER_PATH);
    return 127;
}
