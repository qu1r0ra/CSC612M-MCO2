#include <stdlib.h>

/* Intentional one-byte heap overread; test-cpu-asan must catch this. */
int main(void) {
  volatile unsigned char *bytes = (volatile unsigned char *)malloc(1);
  if (bytes == NULL) {
    return 2;
  }
#pragma warning(push)
#pragma warning(disable : 6200 6385)
  {
    volatile unsigned char value =
        bytes[1]; /* NOLINT(clang-analyzer-security.ArrayBound): ASan probe */
    (void)value;
  }
#pragma warning(pop)
  free((void *)bytes);
  return 0;
}
