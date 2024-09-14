#include <err.h>
#include <unistd.h>
int main(int argc, char ** argv) {
  (void)argc;
  char * p = PROG;
  argv[0] = p;
  execvp(p, argv);
  err(1, "execvp");
}
