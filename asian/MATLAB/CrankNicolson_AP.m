function [price, sol, space, time] = CrankNicolson_AP(s0,sig,r,K,T,n,m)
    Z0 = -(1-exp(-r*T))/(r*T) + K*exp(-r*T)/s0;
    dt = T/n;
    dz = 2*(abs(Z0) + 1)/m;
    d = dt/dz^2;
    
    sol = zeros(n+1,m+1);
    time = 0:dt:T;
    space = -(abs(Z0)+1):dz:(abs(Z0)+1);
    [~,Z0index] = min(abs(space - Z0));
    space = space - space(Z0index) + Z0;
    
    A = zeros(m+1,m+1);
    B = zeros(m+1,m+1);

    A(1,1) = 1;
    A(m+1,m+1) = 1;
    B(1,1) = 1;
    B(m+1,m+1) = 1;

    % boundary conditions
    sol(1,:) = max(space,0);
    sol(:,1) = 0;
    sol(:,m+1) = space(end);

    Q = (1-exp(-r*time))/(r*T);
    alpha = @(h,j) d*sig^2*(Q(h) - space(j))^2/4;

    for i = 2:n+1
        for k = 2:m
            A(k,k-1) = -alpha(i,k);
            A(k,k) = 1 + 2*alpha(i,k);
            A(k,k+1) = -alpha(i,k);

            B(k,k-1) = alpha(i-1,k);
            B(k,k) = 1-2*alpha(i-1,k);
            B(k,k+1) = alpha(i-1,k);
        end
        sol(i,:) = A\(B*sol(i-1,:)');
    end
    price = s0*sol(end,Z0index);
end

