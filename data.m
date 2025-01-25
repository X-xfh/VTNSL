clc;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             clear;clc;               
num_sensor = 16;                 
lambda = 1;
d = 0.25 * lambda;              
J_alpha =1;  
D = (num_sensor - 1)*d;
rmin = 0.62*(D^3/lambda)^0.5;
rmax = 2*D^2/lambda;           
zeta = 0.5;
num_signal =2; 
snr = 10;
K=100;
theta_test = [-8.1 17.3];
distance_test = [3.59 3.88];
train_dir = '';
test_dir = '';


% % % % % 测试集
point = 1;
test_montecarlo = 1000;
X_Test = zeros(point * test_montecarlo, num_sensor, num_sensor,3);
label_theta = repelem(theta_test, test_montecarlo, 1);
label_distance = repelem(distance_test, test_montecarlo, 1);
for snapshot_test = 1:point
    phase_term = phase_difference(theta_test, distance_test, num_sensor, num_signal, d);
    test_data = generate_test_data(num_sensor, num_signal, phase_term, snr,K, test_montecarlo, zeta);
    X_Test(((snapshot_test - 1) * test_montecarlo + 1):(snapshot_test * test_montecarlo),:,:,:) = test_data;

end
save(test_dir,'X_Test', '-v7.3')


% 相位差 
function phase = phase_difference(p_theta, p_distance, p_num_sensor, p_num_signal, p_d)  % phase_difference(doa,r,M,K,d)
phase = zeros(p_num_sensor, p_num_signal);
shift = (p_num_sensor - 1) / 2;
if p_num_signal > 1
    for m = 1:p_num_sensor
        for k = 1:p_num_signal
            phase(m,k) = pi * 2 * (sqrt(p_distance(k)^2 + ((m - shift)* p_d)^2 - 2 * p_distance(k) * (m - shift) * p_d *sind(p_theta(k))) - p_distance(k));
        end
    end
else
    for m = 1:p_num_sensor
        phase(m) = pi * 2 * (sqrt(p_distance^2 + ((m - shift) * p_d)^2 - 2 * p_distance * (m - shift) * p_d * sind(p_theta)) - p_distance);
    end
end
end

function r_output = generate_data(g_num_sensor, g_num_signal, g_snr, g_phase,g_snapshot, g_num_loop, g_zeta, g_i)
steering_matrix = exp(1j * g_phase);
r_output = zeros(g_num_loop, g_num_sensor, g_num_sensor,3);
kb = 0;
for kw = 10:207:207 * g_num_loop
    kw = g_i * 103 + kw;  
    kb = kb + 1;
    rng(kw + 30,'twister');
    if g_num_signal == 1
        rng(kw,'twister');
        s_real = randn(g_snapshot,1);
        rng(kw + 1,'twister');
        s_imag = randn(g_snapshot,1);
        signal = s_real + 1j*s_imag;

    elseif g_num_signal == 2
        rng(kw,'twister');
        s_real = randn(g_snapshot,1);
        rng(kw + 1,'twister');
        s_imag = randn(g_snapshot,1);
        signal_1 = s_real + 1j*s_imag;
        
        rng(kw + 10,'twister');
        s_real = randn(g_snapshot,1);
        rng(kw + 11,'twister');
        s_imag = randn(g_snapshot,1);
        signal_2 = s_real + 1j*s_imag;
    
        signal = [signal_1 signal_2]';

    elseif g_num_signal == 3
        rng(kw,'twister');
        s_real = randn(g_snapshot,1);
        rng(kw + 1,'twister');
        s_imag = randn(g_snapshot,1);
        signal_1 = s_real + 1j*s_imag;
        
        rng(kw + 10,'twister');
        s_real = randn(g_snapshot,1);
        rng(kw + 11,'twister');
        s_imag = randn(g_snapshot,1);
        signal_2 = s_real + 1j*s_imag;

        s_real = randn(g_snapshot,1);
        rng(kw + 20,'twister');
        s_imag = randn(g_snapshot,1);
        signal_3 = s_real + 1j*s_imag;

        signal = [signal_1 signal_2 signal_3]';

     else g_num_signal == 4;
        rng(kw,'twister');
        s_real = randn(g_snapshot,1);
        rng(kw + 1,'twister');
        s_imag = randn(g_snapshot,1);
        signal_1 = s_real + 1j*s_imag;
        
        rng(kw + 10,'twister');
        s_real = randn(g_snapshot,1);
        rng(kw + 11,'twister');
        s_imag = randn(g_snapshot,1);
        signal_2 = s_real + 1j*s_imag;

        s_real = randn(g_snapshot,1);
        rng(kw + 20,'twister');
        s_imag = randn(g_snapshot,1);
        signal_3 = s_real + 1j*s_imag;

        s_real = randn(g_snapshot,1);
        rng(kw + 30,'twister');
        s_imag = randn(g_snapshot,1);
        signal_4 = s_real + 1j*s_imag;

        signal = [signal_1 signal_2 signal_3 signal_4]';
    end        


    g_sigma = 1 / (10^(g_snr / 10));
    Q1 = zeros(g_num_sensor);
    for qi = 1:g_num_sensor
        for qk = 1:g_num_sensor
            Q1(qi,qk) = g_sigma * exp( -(qi-qk)^2 * g_zeta );
        end
    end
    rng(kw+20,'twister');
    w_real = randn(g_num_sensor,g_snapshot);
    rng(kw+21,'twister');
    w_imag = randn(g_num_sensor,g_snapshot);
    w_noise = w_real + 1i*w_imag;
    
    x_signal = steering_matrix * signal + sqrtm(Q1) *  w_noise;
    rr = x_signal * x_signal' / g_snapshot;
    r_output(kb,:,:,1) = real(rr);
    r_output(kb,:,:,2) = imag(rr);
    r_output(kb,:,:,3) = angle(rr);

end
end

function r_output = generate_test_data(g_num_sensor, g_num_signal, g_phase, g_snr,g_snapshot, g_num_loop, g_zeta)
steering_matrix = exp(1j * g_phase);
r_output = zeros(g_num_loop, g_num_sensor, g_num_sensor,3);
kb = 0;
for kw = 10:305:305 * g_num_loop
    kb = kb + 1;
    if g_num_signal == 2 
        rng(kw,'twister');
        s_real = randn(g_snapshot,1);
        rng(kw + 1,'twister');
        s_imag = randn(g_snapshot,1);
        signal_1 = s_real + 1j*s_imag;
        
        rng(kw + 10,'twister');
        s_real = randn(g_snapshot,1);
        rng(kw + 11,'twister');
        s_imag = randn(g_snapshot,1);
        signal_2 = s_real + 1j*s_imag;

        signal = [signal_1 signal_2]';

    elseif g_num_signal == 3
        rng(kw,'twister');
        s_real = randn(g_snapshot,1);
        rng(kw + 1,'twister');
        s_imag = randn(g_snapshot,1);
        signal_1 = s_real + 1j*s_imag;
        
        rng(kw + 10,'twister');
        s_real = randn(g_snapshot,1);
        rng(kw + 11,'twister');
        s_imag = randn(g_snapshot,1);
        signal_2 = s_real + 1j*s_imag;

        s_real = randn(g_snapshot,1);
        rng(kw + 20,'twister');
        s_imag = randn(g_snapshot,1);
        signal_3 = s_real + 1j*s_imag;

        signal = [signal_1 signal_2 signal_3]';

    elseif g_num_signal == 4
        rng(kw,'twister');
        s_real = randn(g_snapshot,1);
        rng(kw + 1,'twister');
        s_imag = randn(g_snapshot,1);
        signal_1 = s_real + 1j*s_imag;
        
        rng(kw + 10,'twister');
        s_real = randn(g_snapshot,1);
        rng(kw + 11,'twister');
        s_imag = randn(g_snapshot,1);
        signal_2 = s_real + 1j*s_imag;

        s_real = randn(g_snapshot,1);
        rng(kw + 20,'twister');
        s_imag = randn(g_snapshot,1);
        signal_3 = s_real + 1j*s_imag;

        s_real = randn(g_snapshot,1);
        rng(kw + 30,'twister');
        s_imag = randn(g_snapshot,1);
        signal_4 = s_real + 1j*s_imag;

        signal = [signal_1 signal_2 signal_3 signal_4]';

    else
        rng(kw,'twister');
        s_real = randn(g_snapshot,1);
        rng(kw + 1,'twister');
        s_imag = randn(g_snapshot,1);
        signal = s_real + 1j*s_imag;
    end
    
    g_sigma = 1 / (10^(g_snr / 10));
    Q1 = zeros(g_num_sensor);
    for qi = 1:g_num_sensor
        for qk = 1:g_num_sensor
            Q1(qi,qk) = g_sigma * exp( -(qi-qk)^2 * g_zeta );
            %Q2(qi,qk) = g_sigma * exp( -abs(qi-qk) * g_zeta );
        end
    end
   
    rng(kw+20,'twister');
    w_real = randn(g_num_sensor,g_snapshot);
    rng(kw+21,'twister');
    w_imag = randn(g_num_sensor,g_snapshot);
    w_noise = w_real + 1i*w_imag;
    
    x_signal = steering_matrix * signal +  w_noise;
    rr = x_signal * x_signal' / g_snapshot;
    r_output(kb,:,:,1) = real(rr);
    r_output(kb,:,:,2) = imag(rr);
    r_output(kb,:,:,3) = angle(rr);

end
end


