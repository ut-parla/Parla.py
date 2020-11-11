#include "Python.h"

// Some magic preprocessor tricks
#define __STRING__(n) #n
#define _STRING(n) __STRING__(n)
#define __CONCAT__(a, b) a##b
#define _CONCAT(a,b) __CONCAT__(a,b)

static PyMethodDef methods[] =
{
	{ NULL, NULL, 0, NULL}
};

static struct PyModuleDef module =
{
	PyModuleDef_HEAD_INIT,
	_STRING(MODULE_NAME),
	"A minimal module to load to fill up the dlopen cache: " _STRING(MODULE_NAME), /* Doc string */
	-1, /* Size of per-interpreter state or -1 */
	methods
};

PyMODINIT_FUNC
_CONCAT(PyInit_, MODULE_NAME)(void)
{
	return PyModule_Create(&module);
}
