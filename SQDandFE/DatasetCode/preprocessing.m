function wav = preprocessing(x)

% mod_A = abs(x);
% normalized_mod_A = normalize(mod_A, 'range');
% wav = normalized_mod_A .* x ./ mod_A;

wav = normalize(x);