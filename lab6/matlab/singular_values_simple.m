% Простая версия: вычисление множества сингулярных чисел
% Явная формула функции σ(ω) прописана вручную

clear; close all; clc;

%% Параметры системы (вариант 13)
A = [0 1; 4 0];
B = [2; 6];
Bf = [0; 2];
Q = eye(2);

%% H∞-синтез
P0 = care(A, B, Q, 1);  % Начальное приближение из LQR
gamma = 0.1;  % Минимальное γ для варианта 13

% Итерационное решение уравнения Риккати
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

fprintf('H∞-синтез завершён\n');
fprintf('K = [%.2f, %.2f]\n\n', K(1), K(2));

%% Извлечение параметров для формулы
a11 = Acl(1, 1);
a12 = Acl(1, 2);
lambda = eig(Acl);
lambda1 = lambda(1);
lambda2 = lambda(2);

fprintf('Параметры для формулы σ(ω):\n');
fprintf('  a11 = %.6f\n', a11);
fprintf('  a12 = %.6f\n', a12);
fprintf('  λ₁ = %.6f\n', lambda1);
fprintf('  λ₂ = %.6f\n\n', lambda2);

%% ЯВНАЯ ФОРМУЛА σ(ω) - ПРОПИСАНА ВРУЧНУЮ
% Согласно определению:
% σᵢ[W(jω)] = √(λᵢ[W^T(-jω)W(jω)]), i = 1, n
% где λᵢ – собственное число
%
% Для нашей системы:
% W(jω) = G(jω) = (jωI - A_cl)^(-1) B_f
% W^T(-jω)W(jω) = G^T(-jω)G(jω) = G^H(jω)G(jω)
%
% Для матрицы размера 2×1 (вектор):
% G^H(jω)G(jω) - скаляр, единственное собственное значение
% σ(ω) = √(λ[G^H(jω)G(jω)]) = √(G^H(jω)G(jω)) = ||G(jω)||₂

fprintf('========================================\n');
fprintf('ЯВНАЯ ФОРМУЛА σ(ω) ЧЕРЕЗ СОБСТВЕННЫЕ ЗНАЧЕНИЯ:\n');
fprintf('========================================\n');
fprintf('σ(ω) = √(λ[G^T(-jω)G(jω)])\n');
fprintf('где G(jω) = (jωI - A_cl)^(-1) B_f\n');
fprintf('      G^T(-jω)G(jω) = G^H(jω)G(jω) - эрмитова матрица\n');
fprintf('      λ - собственное число матрицы G^H(jω)G(jω)\n\n');

% Вычисляем константы для упрощения
a12_sq = a12^2;
a11_sq = a11^2;
lambda1_sq = lambda1^2;
lambda2_sq = lambda2^2;

fprintf('После упрощения для B_f = [0; 2]:\n');
fprintf('σ(ω) = 2 * sqrt(%.6f + ω² + %.6f) / sqrt((ω² + %.6f) * (ω² + %.6f))\n', ...
        a12_sq, a11_sq, lambda1_sq, lambda2_sq);
fprintf('\n');

%% Функция σ(ω) - вычисляется через собственные значения W^T(-jω)W(jω)
% Явная реализация формулы: σ(ω) = √(λ[G^H(jω)G(jω)])
% Согласно определению: σᵢ[W(jω)] = √(λᵢ[W^T(-jω)W(jω)])

% Вычисление на сетке частот
omega = logspace(-2, 2, 1000);  % от 0.01 до 100 рад/с
sigma_values = zeros(size(omega));

for i = 1:length(omega)
    w = omega(i);
    
    % Шаг 1: W(jω) = G(jω) = (jωI - A_cl)^(-1) B_f
    jwI_minus_Acl = 1j * w * eye(size(Acl)) - Acl;
    G_jw = jwI_minus_Acl \ Bf;
    
    % Шаг 2: W^T(-jω)W(jω) = G^T(-jω)G(jω) = G^H(jω)G(jω)
    % Для вектора размера 2×1: G^H(jω)G(jω) - скаляр (1×1 матрица)
    G_H_G = G_jw' * G_jw;  % Эрмитово сопряжённое произведение
    
    % Шаг 3: Находим собственное значение λ[G^H(jω)G(jω)]
    % Для скаляра: собственное значение = сам скаляр
    lambda_val = G_H_G;
    
    % Шаг 4: σ(ω) = √(λ[G^H(jω)G(jω)])
    sigma_values(i) = sqrt(real(lambda_val));  % Берем вещественную часть
end

%% H∞-норма
[hinf_norm, idx] = max(sigma_values);
omega_max = omega(idx);

fprintf('H∞-норма: ||G||_∞ = %.6f при ω = %.6f рад/с\n\n', hinf_norm, omega_max);

%% График
figure('Position', [100, 100, 800, 600]);
semilogx(omega, sigma_values, 'b-', 'LineWidth', 2);
hold on;
yline(hinf_norm, 'g--', 'LineWidth', 1.5);
xlabel('ω, рад/с', 'FontSize', 12);
ylabel('σ(ω)', 'FontSize', 12);
title('Множество сингулярных чисел σ(ω)', 'FontSize', 12);
grid on;
legend('σ(ω)', sprintf('||G||_∞ = %.3f', hinf_norm), 'Location', 'best');

% Сохранение
output_dir = '../images/task6';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
saveas(gcf, fullfile(output_dir, 'singular_values_matlab.png'), 'png');

fprintf('Готово! График сохранён.\n');

