% Deg_RCC_plots.m  -- auto-generated
hyst = readmatrix('Deg_RCC_hysteresis.csv');
env  = readmatrix('Deg_RCC_envelope.csv');
d = hyst(:,1); V = hyst(:,2);
ed = env(:,1); eV = env(:,2); drift = env(:,3); K = env(:,4);

figure; plot(d, V, 'r', 'LineWidth', 1);
xlabel('Lateral Displacement (mm)'); ylabel('Lateral Load (kN)');
title('Hysteretic Response (Degraded RCC)'); grid on;

figure; plot(ed, eV, 'b-o', 'MarkerSize', 4);
xlabel('Lateral Displacement (mm)'); ylabel('Lateral Load (kN)');
title('Capacity Envelope (Degraded RCC)'); grid on;

[~,idx] = sort(abs(ed));
figure; plot(abs(ed(idx)), K(idx), 'g-s', 'MarkerSize', 4);
xlabel('Lateral Displacement (mm)'); ylabel('Stiffness (kN/mm)');
title('Stiffness Degradation (Degraded RCC)'); grid on;

pos = ed > 0;
figure; plot(drift(pos), eV(pos), 'm-o', 'MarkerSize', 4);
xlabel('Storey Drift (%)'); ylabel('Lateral Load (kN)');
title('Capacity Curve (Degraded RCC)'); grid on;
