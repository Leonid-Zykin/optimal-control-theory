% Вычисление множества сингулярных чисел для замкнутой системы H∞-регулятора
% Вариант 13

clear; close all; clc;

%% Параметры системы
A = [0 1; 4 0];
B = [2; 6];
Bf = [0; 2];
Q = eye(2);

%% H∞-синтез: поиск минимального γ
fprintf('H∞-синтез: поиск минимального γ\n');
fprintf('================================\n');

% Начальное приближение из LQR
P0 = care(A, B, Q, 1);

% Поиск минимального γ
gamma_candidates = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0];
gamma_min = [];
P_final = [];
K_final = [];
Acl_final = [];

for gamma = gamma_candidates
    P = P0;
    for iter = 1:40
        Qeff = Q + (1/gamma^2) * (P * Bf * Bf' * P);
        P_next = care(A, B, Qeff, 1);
        if norm(P_next - P, 'fro') < 1e-10
            P = P_next;
            break;
        end
        P = P_next;
    end
    
    K = -B' * P;
    Acl = A + B * K;
    eigvals = eig(Acl);
    
    if all(real(eigvals) < -1e-6)
        gamma_min = gamma;
        P_final = P;
        K_final = K;
        Acl_final = Acl;
        fprintf('Найдено γ_min = %.6f\n', gamma_min);
        break;
    end
end

if isempty(gamma_min)
    error('Не удалось найти стабилизирующее γ');
end

fprintf('Матрица усиления K = [%.2f, %.2f]\n', K_final(1), K_final(2));
fprintf('Собственные значения A_cl: %.6f, %.6f\n\n', eig(Acl_final));

%% Вычисление элементов матрицы и собственных значений
a11 = Acl_final(1, 1);
a12 = Acl_final(1, 2);
a21 = Acl_final(2, 1);
a22 = Acl_final(2, 2);

lambda = eig(Acl_final);
lambda1 = lambda(1);
lambda2 = lambda(2);

fprintf('Элементы матрицы A_cl:\n');
fprintf('  a11 = %.6f\n', a11);
fprintf('  a12 = %.6f\n', a12);
fprintf('  a21 = %.6f\n', a21);
fprintf('  a22 = %.6f\n', a22);
fprintf('\nСобственные значения:\n');
fprintf('  λ₁ = %.6f\n', lambda1);
fprintf('  λ₂ = %.6f\n\n', lambda2);

%% ЯВНАЯ ФОРМУЛА ДЛЯ σ(ω)
% σ(ω) = ||(jωI - A_cl)^(-1) B_f||_2
%      = ||adj(jωI - A_cl) B_f||_2 / |det(jωI - A_cl)|
%
% Для B_f = [0; 2]:
% adj(jωI - A_cl) B_f = [2*a12; 2*(jω - a11)]
%
% ||adj(jωI - A_cl) B_f||_2 = 2*sqrt(a12^2 + ω^2 + a11^2)
%
% det(jωI - A_cl) = (jω - λ₁)(jω - λ₂)
% |det(jωI - A_cl)| = sqrt((ω^2 + λ₁^2)(ω^2 + λ₂^2))
%
% ИТОГОВАЯ ФОРМУЛА:
% σ(ω) = 2*sqrt(a12^2 + ω^2 + a11^2) / sqrt((ω^2 + λ₁^2)(ω^2 + λ₂^2))

fprintf('========================================\n');
fprintf('ЯВНАЯ ФОРМУЛА ДЛЯ σ(ω):\n');
fprintf('========================================\n');
fprintf('σ(ω) = 2*sqrt(a12² + ω² + a11²) / sqrt((ω² + λ₁²)(ω² + λ₂²))\n');
fprintf('\nгде:\n');
fprintf('  a11 = %.6f\n', a11);
fprintf('  a12 = %.6f\n', a12);
fprintf('  λ₁ = %.6f\n', lambda1);
fprintf('  λ₂ = %.6f\n\n', lambda2);

%% Функция для вычисления σ(ω) по явной формуле
sigma_func = @(omega) 2 * sqrt(a12^2 + omega.^2 + a11^2) ./ ...
                      sqrt((omega.^2 + lambda1^2) .* (omega.^2 + lambda2^2));

%% Вычисление на сетке частот
omega = logspace(-2, 2, 1000);  % Частоты от 0.01 до 100 рад/с
sigma_omega = sigma_func(omega);

%% Нахождение H∞-нормы
[hinf_norm, idx_max] = max(sigma_omega);
omega_max = omega(idx_max);

fprintf('H∞-норма: ||G||_∞ = %.6f при ω = %.6f рад/с\n\n', hinf_norm, omega_max);

%% Построение графика
figure('Position', [100, 100, 800, 600]);
semilogx(omega, sigma_omega, 'b-', 'LineWidth', 2);
hold on;
yline(hinf_norm, 'g--', 'LineWidth', 1.5, 'DisplayName', sprintf('||G||_∞ = %.3f', hinf_norm));
xline(omega_max, 'g--', 'LineWidth', 1.5, 'Alpha', 0.5);
xlabel('ω, рад/с', 'FontSize', 12);
ylabel('σ(ω)', 'FontSize', 12);
title('Множество сингулярных чисел: σ(ω) = 2\sqrt{a_{12}^2 + ω^2 + a_{11}^2} / \sqrt{(ω^2 + λ_1^2)(ω^2 + λ_2^2)}', ...
      'FontSize', 12);
grid on;
legend('σ(ω)', 'Location', 'best', 'FontSize', 10);

% Сохранение графика
output_dir = '../images/task6';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
saveas(gcf, fullfile(output_dir, 'singular_values_matlab.png'), 'png');
fprintf('График сохранён: %s\n', fullfile(output_dir, 'singular_values_matlab.png'));

%% Сохранение данных
data_file = fullfile(output_dir, 'singular_values_matlab_data.txt');
fid = fopen(data_file, 'w');
fprintf(fid, '# Множество сингулярных чисел передаточной функции G(s) = (sI - A_cl)^(-1) B_f\n');
fprintf(fid, '# Формат: ω (рад/с) | σ(ω)\n');
fprintf(fid, '# H∞-норма: ||G||_∞ = %.6f при ω = %.6f рад/с\n', hinf_norm, omega_max);
fprintf(fid, '# γ_min = %.6f\n', gamma_min);
fprintf(fid, '# K = [%.6f, %.6f]\n', K_final(1), K_final(2));
fprintf(fid, '#\n');
fprintf(fid, '# Явная формула:\n');
fprintf(fid, '# σ(ω) = 2*sqrt(a12² + ω² + a11²) / sqrt((ω² + λ₁²)(ω² + λ₂²))\n');
fprintf(fid, '# где a11 = %.6f, a12 = %.6f, λ₁ = %.6f, λ₂ = %.6f\n', ...
        a11, a12, lambda1, lambda2);
fprintf(fid, '#\n');
for i = 1:length(omega)
    fprintf(fid, '%.6e  %.6e\n', omega(i), sigma_omega(i));
end
fclose(fid);
fprintf('Данные сохранены: %s\n', data_file);

%% Вывод итоговой информации
fprintf('\n========================================\n');
fprintf('ИТОГОВАЯ ИНФОРМАЦИЯ:\n');
fprintf('========================================\n');
fprintf('Множество сингулярных чисел:\n');
fprintf('  σ(G) = {σ(ω) : ω ∈ ℝ}\n');
fprintf('\nФункция сингулярного числа:\n');
fprintf('  σ(ω) = 2*sqrt(%.6f + ω² + %.6f) / sqrt((ω² + %.6f)(ω² + %.6f))\n', ...
        a12^2, a11^2, lambda1^2, lambda2^2);
fprintf('\nH∞-норма (максимум множества):\n');
fprintf('  ||G||_∞ = max_{ω∈ℝ} σ(ω) = %.6f\n', hinf_norm);
fprintf('  достигается при ω = %.6f рад/с\n', omega_max);

