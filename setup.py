#!/usr/bin/env python

from distutils.core import setup, Extension

cache_filler_modules = []
for i in range(128):
      module_name = 'cache_filler_' + str(i)
      module = Extension("parla." + module_name,
                          define_macros = [('MODULE_NAME', module_name)],
                          sources = ['cache_filler.c'])
      cache_filler_modules.append(module)

setup(name = "parla",
      version = "0.1",
      description = "The parla Python frontend.",
      packages = ['parla'],
      ext_modules=cache_filler_modules,
      )
