using System;
using UnityEngine;

namespace DL
{
    public enum ActivationFunction
    {
        ReLu,
        Tanh,
        Softmax,
        Linear
    }
    
    public abstract class Layer 
    {
        public Array Output;
        public Array DInput;

        //TODO: in final version weights and biases should be protected not public
        public Array _weights;
        public Array _biases;
    
        protected readonly bool _isFirstLayer;
        
        protected bool _forwardInitialized;
        protected bool _backwardInitialized;
    
        protected readonly ComputeShader _shader;
    
        protected ComputeBuffer _inputBuffer;
        protected ComputeBuffer _outputBuffer;
        protected ComputeBuffer _weightsBuffer;
        protected ComputeBuffer _biasesBuffer;
    
        // forward only variables
        protected int _kernelHandleForward;

        //backwards variables
        protected ComputeBuffer _dValuesBuffer;
        protected ComputeBuffer _dInputsBuffer;
    
        protected int _kernelHandleInputsBackward;
        protected int _kernelHandleWeightsBiasesBackward;
    
        // Adam optimizer
        protected ComputeBuffer _weightsMomentumBuffer;
        protected ComputeBuffer _weightsCacheBuffer;
        protected ComputeBuffer _biasesMomentumBuffer;
        protected ComputeBuffer _biasesCacheBuffer;
    
        // cashed variables
        protected readonly int _currentLearningRateID;

        protected Layer(ComputeShader shader, bool isFirstLayer)
        {
            _isFirstLayer = isFirstLayer;
            _shader = shader;
            _currentLearningRateID = Shader.PropertyToID("current_learning_rate");
        }
    
        public abstract void Forward(Array input);

        public abstract void Backward(Array dValue, float currentLearningRate);

        protected abstract void InitializeForwardBuffers(Array inputs);

        protected abstract void InitializeBackwardsBuffers(Array dValue);
    
        public void SetOptimizerVariables(float beta1, float beta2, float epsilon)
        {
            //TODO: can set set different optimizers based on a condition
            _shader.SetFloat("beta_1", beta1);
            _shader.SetFloat("beta_2", beta2);
            _shader.SetFloat("epsilon", epsilon);
            
            _shader.SetFloat("negated_beta_1", 1 - beta1);
            _shader.SetFloat("negated_beta_2", 1 - beta2);
        }
    
        public virtual void CopyLayer(Layer otherLayer)
        {
            _weightsBuffer.GetData(otherLayer._weights);
            otherLayer._weightsBuffer.SetData(otherLayer._weights);

            _biasesBuffer.GetData(otherLayer._biases);
            otherLayer._biasesBuffer.SetData(otherLayer._biases);
        }
    
        public virtual void Dispose()
        {
            _weightsMomentumBuffer?.Dispose();
            _weightsCacheBuffer?.Dispose();
            _biasesMomentumBuffer?.Dispose();
            _biasesCacheBuffer?.Dispose();

            _inputBuffer?.Dispose();
            _outputBuffer?.Dispose();
            _weightsBuffer?.Dispose();
            _biasesBuffer?.Dispose();

            _dValuesBuffer?.Dispose();
            _dInputsBuffer?.Dispose();
        }
    }
}
