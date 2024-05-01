function x = l2_weights(A,b,C,Nsample)

if C == 0
    x = pinv(A)*b;
else
    if size(A,2)<Nsample
        x = (eye(size(A,2))/C+A'*A) \ A'*b;
    else
        x = A'*((eye(size(A,1))/C+A*A') \ b);
    end
end

end
% toc

