using UnityEngine;

namespace NN
{
    public abstract class NetworkLoss
    {
        public float[,] DInputs;
        public float[] SampleLosses;
        
        protected readonly ComputeShader Shader;
        protected int KernelHandleForwardLoss;
        private int _threadGroupXForward;
        private ComputeBuffer _yTrueBuffer;
        private ComputeBuffer _sampleLossesBuffer;
        private ComputeBuffer _outputBuffer;

        private bool _forwardInit;

        // backward params
        protected int KernelHandleBackwardLoss;
        private int _threadGroupXBackward;
        private int _threadGroupYBackward;
        private ComputeBuffer _dInputsLossBuffer;

        private bool _backwardInit;

        protected NetworkLoss(ComputeShader shader)
        {
            Shader = shader;
            _forwardInit = false;
            _backwardInit = false;
        }
        
        public virtual void Calculate(float[,] output, float[,] yTrue)
        {
            if (!_forwardInit)
            {
                InitBuffers(output, yTrue);

                SampleLosses = new float[yTrue.GetLength(0)];

                Shader.GetKernelThreadGroupSizes(KernelHandleForwardLoss, out var threadSizeX, out _, out _);
                _threadGroupXForward = Mathf.CeilToInt(yTrue.GetLength(0) / (float)threadSizeX);

                _sampleLossesBuffer = new ComputeBuffer(yTrue.GetLength(0), sizeof(float));
                Shader.SetBuffer(KernelHandleForwardLoss, "sample_losses", _sampleLossesBuffer);

                Shader.SetBuffer(KernelHandleForwardLoss, "output", _outputBuffer);
                Shader.SetBuffer(KernelHandleForwardLoss, "y_true", _yTrueBuffer);

                _forwardInit = true;
            }

            _outputBuffer.SetData(output);
            _yTrueBuffer.SetData(yTrue);
            
            //TODO: this might be faster if done in a unity job, since it is just using the x group
            Shader.Dispatch(KernelHandleForwardLoss, _threadGroupXForward, 1, 1);
            _sampleLossesBuffer.GetData(SampleLosses);
        }

        public virtual void Backward(float[,] output, float[,] yTrue)
        {
            if (!_backwardInit)
            {
                InitBuffers(output, yTrue);

                DInputs = new float[output.GetLength(0), output.GetLength(1)];

                Shader.GetKernelThreadGroupSizes(KernelHandleBackwardLoss, out var threadSizeX, out var threadSizeY,
                    out _);
                _threadGroupXBackward = Mathf.CeilToInt(output.GetLength(0) / (float)threadSizeX);
                _threadGroupYBackward = Mathf.CeilToInt(output.GetLength(1) / (float)threadSizeY);

                _dInputsLossBuffer = new ComputeBuffer(output.Length, sizeof(float));
                Shader.SetBuffer(KernelHandleBackwardLoss, "d_inputs_loss", _dInputsLossBuffer);

                Shader.SetBuffer(KernelHandleBackwardLoss, "output", _outputBuffer);
                Shader.SetBuffer(KernelHandleBackwardLoss, "y_true", _yTrueBuffer);

                _backwardInit = true;
            }

            _outputBuffer.SetData(output);
            _yTrueBuffer.SetData(yTrue);
            Shader.Dispatch(KernelHandleBackwardLoss, _threadGroupXBackward, _threadGroupYBackward, 1);
            _dInputsLossBuffer.GetData(DInputs);
        }
        
        private void InitBuffers(float[,] output, float[,] yTrue)
        {
            if (_forwardInit || _backwardInit)
                return;

            Shader.SetInt("input_column_size", output.GetLength(0));
            Shader.SetInt("weights_row_size", output.GetLength(1));

            _outputBuffer = new ComputeBuffer(output.Length, sizeof(float));
            _yTrueBuffer = new ComputeBuffer(yTrue.Length, sizeof(float));
        }
        
        public virtual void Dispose()
        {
            _yTrueBuffer?.Dispose();
            _sampleLossesBuffer?.Dispose();
            _outputBuffer?.Dispose();
            _dInputsLossBuffer?.Dispose();
        }
    }
    
    public class MeanSquaredError : NetworkLoss
    {
        public MeanSquaredError(ComputeShader shader) : base(shader)
        {
            KernelHandleForwardLoss = Shader.FindKernel("forward_pass_MSE_loss");
            KernelHandleBackwardLoss = Shader.FindKernel("backwards_pass_MSE_loss");
        }
    }

    public class MeanSquaredErrorPrioritized : NetworkLoss
    {
        private ComputeBuffer _sampleWeightsBuffer;
        private bool _inLearningCycle;
        public MeanSquaredErrorPrioritized(ComputeShader shader) : base(shader)
        {
            KernelHandleForwardLoss = Shader.FindKernel("forward_pass_MSE_prioritized_loss");
            KernelHandleBackwardLoss = Shader.FindKernel("backwards_pass_MSE_prioritized_loss");
        }

        public void SetLossExternalParameters(float[] parameters)
        {
            if (_sampleWeightsBuffer == null)
            {
                _sampleWeightsBuffer = new ComputeBuffer(parameters.Length, sizeof(float));
                //Shader.SetBuffer(KernelHandleForwardLoss, "sample_weights", _sampleWeightsBuffer);
                Shader.SetBuffer(KernelHandleBackwardLoss, "sample_weights", _sampleWeightsBuffer);
            }
            
            _sampleWeightsBuffer.SetData(parameters);
            _inLearningCycle = true;
        }

        public override void Dispose()
        {
            base.Dispose();
            _sampleWeightsBuffer?.Dispose();
        }
    }
}