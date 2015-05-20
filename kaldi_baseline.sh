combine-score-path ark:- "ark:gunzip -c data/nn_post/train.lab_1000_0.16.gz |" | weight-score-path ark:- -1 ark:- | best-score-path ark:- ark:- | split-score-path ark:- ark:/dev/null ark,t:-
