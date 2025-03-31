
classdef customRegressionLayer < nnet.layer.RegressionLayer
    properties
        QDELTA
    end
    
    methods
        function layer = customRegressionLayer(name, QDELTA)
            layer.Name = name;
            layer.Description = 'layer using custom loss function';
            layer.QDELTA = QDELTA;
        end

        function loss = forwardLoss(layer,output, target)
            loss = customLoss(output, target, layer.QDELTA);
        end

    end
end
