function s = Costas(NumberSamples, fs, A, fcmin, NumHop) 
tsub = (1:ceil(NumberSamples/length(NumHop)))/fs;
f = NumHop*fcmin;
phi0 = 2*pi*rand(1) - pi;
for k = 1:length(f)
    s1(:,k) = A*exp(1j*(2*pi*f(k)*tsub+phi0));
end
[u,v] = size(s1);
s = reshape(s1,[1, u*v]);
end
