compute-wer "ark:trim-path ark:data/test.lab ark:- |" "ark:split-path-score ark:data/nn_post/test.tags ark:/dev/null ark:- | trim-path ark:- ark:- |"
