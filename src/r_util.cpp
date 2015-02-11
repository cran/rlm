#include <math.h>

#include "r_util.h"

typedef union
{
    double value;
    unsigned int word[2];
} ieee_double;

static const int hw = 1;
static const int lw = 0;

bool R_UTIL::is_na(const double x)
{
    if (isnan(x)) {
        ieee_double y;
        y.value = x;
        return (y.word[lw] == 1954);
    }
    return false;
}

void R_UTIL::set_na(double &x) {
    volatile ieee_double y;
    y.word[hw] = 0x7ff00000;
    y.word[lw] = 1954;
    x = y.value;
}
