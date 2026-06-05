% CFRP_full_plots.m -- auto-generated
hyst = readmatrix('CFRP_full_hysteresis.csv');
env  = readmatrix('CFRP_full_envelope.csv');
d = hyst(:,1); V = hyst(:,2);
ed = env(:,1); eV = env(:,2); drift = env(:,3); K = env(:,4);

figure; plot(d, V, 'r', 'LineWidth', 1); grid on;
xlabel('Lateral Displacement (mm)'); ylabel('Lateral Load (kN)');
title('Hysteretic Response (Full CFRP Retrofit)');

figure; plot(ed, eV, 'b-o', 'MarkerSize', 4); grid on;
xlabel('Lateral Displacement (mm)'); ylabel('Lateral Load (kN)');
title('Capacity Envelope (Full CFRP Retrofit)');

[~,idx] = sort(abs(ed));
figure; plot(abs(ed(idx)), K(idx), 'g-s', 'MarkerSize', 4); grid on;
xlabel('Lateral Displacement (mm)'); ylabel('Stiffness (kN/mm)');
title('Stiffness Degradation (Full CFRP Retrofit)');

pos = ed > 0;
figure; plot(ed(pos), eV(pos), 'm-o', 'MarkerSize', 4); grid on;
xlabel('Top-Storey Displacement (mm)'); ylabel('Lateral Load (kN)');
title('Capacity Curve (Full CFRP Retrofit)');
