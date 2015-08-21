#!/bin/bash

cat parallel | parallel --workdir ~/StructureNN/ -j 2 -S :
