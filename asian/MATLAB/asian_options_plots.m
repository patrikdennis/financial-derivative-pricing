%% solve_asian_options.m
% This script prices Asian call and put options using various methods.
% It plots the option prices as functions of the initial price S0 and 
% volatility sigma, verifies put–call parity, and compares the efficiency 
% of the finite difference and control variate Monte Carlo methods.

clear; close all; clc;

%% Parameters
r    = 0.05;   % risk-free interest rate
K    = 100;    % strike price
T    = 1;      % time to maturity in years

% Ranges for the initial stock price and volatility
S0_list    = [80, 90, 100, 110, 120]; 
sigma_list = [0.1, 0.2, 0.3, 0.4, 0.5];

% Finite Difference (Crank-Nicolson) parameters
n_FD = 100;   % number of time steps
m_FD = 100;   % number of space steps

% Monte Carlo parameters
n_MC = 50;    % number of time discretization steps for stock paths
N_MC = 10000; % number of Monte Carlo paths (per simulation)

% Preallocate result arrays (rows: S0 values, columns: sigma values)
numS0   = length(S0_list);
numSig  = length(sigma_list);

% Prices using closed-form geometric formulas (for reference & control variate)
call_geo = zeros(numS0, numSig);
put_geo  = zeros(numS0, numSig);

% Prices using finite difference (Crank-Nicolson)
call_FD  = zeros(numS0, numSig);
put_FD   = zeros(numS0, numSig);

% Prices using Monte Carlo with control variate
call_MC  = zeros(numS0, numSig);
put_MC   = zeros(numS0, numSig);

% CPU times for FD and MC methods
time_FD  = zeros(numS0, numSig);
time_MC  = zeros(numS0, numSig);

%% Loop over S0 and sigma values
for i = 1:numS0
    s0 = S0_list(i);
    for j = 1:numSig
        sig = sigma_list(j);
        
        %--- Closed-form (geometric average) prices ---
        call_geo(i,j) = ClosedFormula_AC_geo(s0, sig, r, K, T);
        put_geo(i,j)  = ClosedFormula_AP_geo(s0, sig, r, K, T);
        
        %--- Finite Difference: Crank-Nicolson method ---
        % Asian Call
        tic;
        [call_price_FD, ~, ~, ~] = CrankNicolson_AC(s0, sig, r, K, T, n_FD, m_FD);
        t1 = toc;
        % Asian Put
        tic;
        [put_price_FD, ~, ~, ~]  = CrankNicolson_AP(s0, sig, r, K, T, n_FD, m_FD);
        t2 = toc;
        time_FD(i,j) = t1 + t2;
        call_FD(i,j) = call_price_FD;
        put_FD(i,j)  = put_price_FD;
        
        %--- Monte Carlo with Control Variate ---
        % Asian Call
        tic;
        [call_price_MC, ~] = MonteCarlo_AC(s0, sig, r, K, T, n_MC, N_MC);
        t3 = toc;
        % Asian Put
        tic;
        [put_price_MC, ~]  = MonteCarlo_AP(s0, sig, r, K, T, n_MC, N_MC);
        t4 = toc;
        time_MC(i,j) = t3 + t4;
        call_MC(i,j) = call_price_MC;
        put_MC(i,j)  = put_price_MC;
    end
end



%% Arithmetic Put–Call Parity Verification and Table

% We'll create a results array with 7 columns:
% [ S0,  sigma,  LHS_MC,   LHS_FD,   Theoretical,   Err_MC,   Err_FD ]
results = zeros(numS0*numSig, 7);

rowCount = 1;
for i = 1:numS0
    s0 = S0_list(i);

    % The new theoretical parity formula (arithmetic version) at t=0:
    %   Call(0) - Put(0) = e^{-rT} * ( ((exp(rT)-1)/(rT))*s0 - K )
    parity_theory = exp(-r*T) * ( ((exp(r*T) - 1)/(r*T))*s0 - K );

    for j = 1:numSig
        sig = sigma_list(j);

        % LHS from Monte Carlo
        lhs_MC = call_MC(i,j) - put_MC(i,j);

        % LHS from Crank–Nicolson
        lhs_FD = call_FD(i,j) - put_FD(i,j);

        % Errors vs. the theoretical formula
        err_MC = lhs_MC - parity_theory;
        err_FD = lhs_FD - parity_theory;

        % Store in the results matrix
        results(rowCount,1) = s0;
        results(rowCount,2) = sig;
        results(rowCount,3) = lhs_MC;
        results(rowCount,4) = lhs_FD;
        results(rowCount,5) = parity_theory;
        results(rowCount,6) = err_MC;
        results(rowCount,7) = err_FD;

        rowCount = rowCount + 1;
    end
end

% Convert to table for clarity
colNames = {'S0','sigma','(Call-Put)MC','(Call-Put)FD','Theory','ErrorMC','ErrorFD'};
resultsTable = array2table(results,'VariableNames',colNames);

disp('Put–Call Parity Comparison for Arithmetic-Average Asian Options');
disp(resultsTable);



%% Example Plots: Arithmetic Put–Call Parity

% 1) Plot vs. S0 for a chosen sigma (e.g., sigma=0.2)
sigma_plot = 0.2;
mask_sigma = (abs(resultsTable.sigma - sigma_plot) < 1e-12); 
% (or use == if your sigma_list is exact decimals)

% Extract rows that match sigma=0.2
temp_sigma = resultsTable(mask_sigma,:);
% Sort by S0 to get a nice monotonic x-axis
temp_sigma = sortrows(temp_sigma, 'S0');

figure('Name','Parity vs S0','NumberTitle','off');
hold on; grid on;
plot(temp_sigma.S0, temp_sigma.("(Call-Put)MC"), 'r-o', 'LineWidth',1.5, 'DisplayName','MC: Call-Put');
plot(temp_sigma.S0, temp_sigma.("(Call-Put)FD"), 'b-o', 'LineWidth',1.5, 'DisplayName','FD: Call-Put');
plot(temp_sigma.S0, temp_sigma.Theory,          'k--','LineWidth',1.5, 'DisplayName','Theory');
xlabel('S_0'); 
ylabel('Difference (Call - Put)');
legend('Location','best');
title(['Arithmetic Put-Call Parity @ \sigma = ' num2str(sigma_plot)]);
hold off;

% 2) Plot vs. sigma for a chosen S0 (e.g., S0=100)
S0_plot = 100;
mask_S0 = (abs(resultsTable.S0 - S0_plot) < 1e-12);

% Extract rows that match S0=100
temp_S0 = resultsTable(mask_S0,:);
% Sort by sigma
temp_S0 = sortrows(temp_S0, 'sigma');

figure('Name','Parity vs sigma','NumberTitle','off');
hold on; grid on;
plot(temp_S0.sigma, temp_S0.("(Call-Put)MC"), 'r-o', 'LineWidth',1.5, 'DisplayName','MC: Call-Put');
plot(temp_S0.sigma, temp_S0.("(Call-Put)FD"), 'b-o', 'LineWidth',1.5, 'DisplayName','FD: Call-Put');
plot(temp_S0.sigma, temp_S0.Theory,          'k--','LineWidth',1.5, 'DisplayName','Theory');
xlabel('\sigma'); 
ylabel('Difference (Call - Put)');
legend('Location','best');
title(['Arithmetic Put-Call Parity @ S_0 = ' num2str(S0_plot)]);
hold off;



%% 1. Plotting the Asian Option Prices as Functions of S0 and sigma

% Plot Asian Call Price (Finite Difference)
figure;
surf(S0_list, sigma_list, call_FD');
xlabel('Initial Stock Price, S0');
ylabel('Volatility, \sigma');
zlabel('Asian Call Price (FD)');
title('Asian Call Price (Finite Difference) vs S0 and \sigma');
colorbar; grid on;

% Plot Asian Put Price (Finite Difference)
figure;
surf(S0_list, sigma_list, put_FD');
xlabel('Initial Stock Price, S0');
ylabel('Volatility, \sigma');
zlabel('Asian Put Price (FD)');
title('Asian Put Price (Finite Difference) vs S0 and \sigma');
colorbar; grid on;

%% 2. Verification of Put-Call Parity
% For geometric Asian options, the theoretical put-call parity is:
%     Call - Put = exp((T/2)*(r - sigma^2/6))*S0 - K

put_call_error = zeros(numS0, numSig);
parity_RHS     = zeros(numS0, numSig);
for i = 1:numS0
    s0 = S0_list(i);
    %S0_bar_disc = (s0 / N_disc) * sum(exp(r * times));
    for j = 1:numSig
        sig = sigma_list(j);
        lhs = call_geo(i,j) - put_geo(i,j);
        %rhs = exp((T/2)*(r - sig^2/6)) * s0 - K;
        % parity_RHS(i,j) = rhs;
        %rhs = exp(-r*T) * (S0_bar_disc - K);
        %parity_RHS(i,j)        = rhs;
        put_call_error(i,j) = abs(lhs - rhs);
    end
end

max_error = max(put_call_error(:));
fprintf('Maximum put-call parity error (geometric closed-form): %e\n', max_error);

%% 3. Comparison: Finite Difference vs. Monte Carlo (Control Variate)
% For a fixed S0 (choose S0 = 100, or the closest value), compare call prices 
% for varying sigma.
idx_S0 = find(S0_list==100);
if isempty(idx_S0)
    [~, idx_S0] = min(abs(S0_list - 100));
end

comparisonTable = table(sigma_list', ...
    call_FD(idx_S0,:)', ...
    call_MC(idx_S0,:)', ...
    abs(call_FD(idx_S0,:) - call_MC(idx_S0,:))', ...
    'VariableNames',{'Sigma','Call_FD','Call_MC','Abs_Difference'});
disp('Comparison of Asian Call Prices for S0 = 100:');
disp(comparisonTable);

%% 4. Efficiency Comparison (CPU Timing)
avg_time_FD = mean(time_FD(:));
avg_time_MC = mean(time_MC(:));
fprintf('Average CPU Time for Finite Difference method: %.4f seconds\n', avg_time_FD);
fprintf('Average CPU Time for Monte Carlo (Control Variate) method: %.4f seconds\n', avg_time_MC);

% Plot CPU times as functions of sigma for S0=100
figure;
plot(sigma_list, time_FD(idx_S0,:), 'bo-', 'LineWidth', 2); hold on;
plot(sigma_list, time_MC(idx_S0,:), 'rs-', 'LineWidth', 2);
xlabel('Volatility, \sigma');
ylabel('CPU Time (seconds)');
legend('Finite Difference', 'Monte Carlo (CV)', 'Location','northwest');
title('CPU Time Comparison for S0 = 100');
grid on;

%% 5. Additional Plot: FD vs MC Price Difference (Call Option)
figure;
plot(sigma_list, call_FD(idx_S0,:) - call_MC(idx_S0,:), 'k*-', 'LineWidth', 2);
xlabel('Volatility, \sigma');
ylabel('Price Difference (FD - MC)');
title('Difference between FD and MC Asian Call Prices for S0 = 100');
grid on;


%%

%% Additional Plots: FDS vs CVMC for Varying S0 and sigma

% --------------------------
% 1) Plot vs. S0 for a chosen sigma
% --------------------------
sigma_plot = 0.2;  % choose a volatility of interest
[~, idx_sigma] = min(abs(sigma_list - sigma_plot));

figure;
hold on;
plot(S0_list, call_FD(:, idx_sigma), 'r-o', 'LineWidth', 1.5);
plot(S0_list, put_FD(:, idx_sigma),  'b-o', 'LineWidth', 1.5);
plot(S0_list, call_MC(:, idx_sigma), 'r--*', 'LineWidth', 1.5);
plot(S0_list, put_MC(:, idx_sigma),  'b--*', 'LineWidth', 1.5);
yline(K, 'k--', 'Strike Price');  % dashed horizontal line at the strike
legend('FDS Call Option','FDS Put Option','CVMC Call Option','CVMC Put Option','Strike Price','Location','best');
xlabel('S_0');
ylabel('Price  (P(0))');
title(['Comparison of CVMC and FDS Methods for Asian Options Varying Initial Price ( \sigma = ' num2str(sigma_list(idx_sigma)) ' )']);
grid on;
hold off;

% --------------------------
% 2) Plot vs. sigma for a chosen S0
% --------------------------
S0_plot = 100;  % choose an S0 of interest
[~, idx_S0] = min(abs(S0_list - S0_plot));

figure;
hold on;
plot(sigma_list, call_FD(idx_S0,:), 'r-o', 'LineWidth', 1.5);
plot(sigma_list, put_FD(idx_S0,:),  'b-o', 'LineWidth', 1.5);
plot(sigma_list, call_MC(idx_S0,:), 'r--*', 'LineWidth', 1.5);
plot(sigma_list, put_MC(idx_S0,:),  'b--*', 'LineWidth', 1.5);
yline(K, 'k--', 'Strike Price');  % dashed horizontal line at the strike
legend('FDS Call Option','FDS Put Option','CVMC Call Option','CVMC Put Option','Strike Price','Location','best');
xlabel('\sigma');
ylabel('Price  (P(0))');
title(['Comparison of CVMC and FDS Methods for Asian Options Varying Volatility ( S_0 = ' num2str(S0_list(idx_S0)) ' )']);
grid on;
hold off;

