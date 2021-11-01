#! /bin/bash

#for f in p1imgs/*.png; do echo "$f"; done
for f in p1imgs/*.png; do mv "$f" "$(echo "$f" | sed s/_/""/)"; done
#for f in p2_imgs/*.jpg; do echo "$f"; done
#for f in p2_imgs/*.jpg; do mv "$f" "$(echo "$f" | sed s/_sat/""/)"; done