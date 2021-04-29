#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

#define UNW_LOCAL_ONLY
#include <libunwind.h>


// Adapted from an example at https://eli.thegreenplace.net/2015/programmatic-access-to-the-call-stack-in-c/
// This approach fails to print out symbols from VECs, but it at least keeps going so we get a partial stack trace.
inline void show_backtrace_with_unwind(void) {
  unw_cursor_t cursor;
  unw_context_t context;

  // Initialize cursor to current frame for local unwinding.
  unw_getcontext(&context);
  unw_init_local(&cursor, &context);

  // Unwind frames one by one, going up the frame stack.
  while (unw_step(&cursor) > 0) {
    unw_word_t offset, pc;
    unw_get_reg(&cursor, UNW_REG_IP, &pc);
    if (pc == 0) {
      break;
    }
    printf("0x%lx:", pc);

    char sym[256];
    if (unw_get_proc_name(&cursor, sym, sizeof(sym), &offset) == 0) {
      printf(" (%s+0x%lx)\n", sym, offset);
    } else {
      printf(" -- error: unable to obtain symbol name for this frame\n");
    }
  }
}

// Just use backtrace directly.
// This stops as soon as it fails to print a given symbol, so we don't even get a partial backtrace in that case.
inline void show_backtrace(void) {
  void *array[10];
  size_t size;

  // get void*'s for all entries on the stack
  size = backtrace(array, 10);

  // print out all the frames to stderr
  backtrace_symbols_fd(array, size, STDERR_FILENO);
}

inline void handler(int sig) {
  fprintf(stderr, "Error: signal %d:\n", sig);
  show_backtrace_with_unwind();
  exit(1);
}

inline void register_handler() {
  signal(SIGSEGV, handler);
}
