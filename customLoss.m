function loss = customLoss(output, target, QDELTA)
% Custom loss combining data-driven MSE and physics-based QDELTA deviation

% Match QDELTA length to current batch/sequence
seqLen = size(target, 3);
QDELTA_trimmed = QDELTA(1:seqLen);

% Convert to dlarray with same shape as target/output
dlQDELTA = dlarray(reshape(QDELTA_trimmed', 1, 1, []));

% Physics-based loss (Qout vs Qdelta)
loss_physics = mean((output - dlQDELTA).^2, 'all');

% Standard MSE loss
loss_data = mean((output - target).^2, 'all');

% Combined loss with weights
lambda1 = 0.5;  % data loss weight
lambda2 = 0.5;  % physics loss weight

loss = lambda1 * loss_data + lambda2 * loss_physics;
end
