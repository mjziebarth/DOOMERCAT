project('doomercat_cppextensions','cpp')

add_project_arguments('--std=c++23', language: 'cpp')

boost_subset = subproject('boost-subset')

sources = ['cpp/src/hotine.cpp', 'cpp/src/functions.cpp',
           'cpp/src/arithmetic.cpp', 'cpp/src/dataset.cpp',
           'cpp/src/ctypesinterface.cpp',
           'cpp/src/linalg.cpp', 'cpp/src/cost_hotine.cpp']

dependencies = [
    dependency('openmp'),
    boost_subset.get_variable('boost_subset')
]

shared_library(
    '_cppextensions',
    sources,
    dependencies : dependencies,
    include_directories : 'cpp/include'
)