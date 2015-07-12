#!/bin/bash

cat parallel | parallel --workdir ~/StructureNN/ -j 2 -S Hormes -S Tigris -S Eupharates
