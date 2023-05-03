using UnityEngine;

namespace NN
{
    // This class uses a single thread, this is because DQN uses small batch sizes, the matrix parameters are of size 32x10
    // For smaller sizes single thread is faster, if matrices get bigger consider jobs first,
    // and for larger values use the old class with compute shaders
    public abstract class NetworkLoss
    {
        public float[,] DInputs;
        public float[] SampleLosses;
        protected int _columnSize;
        protected int _rowSize;

        protected readonly ComputeShader Shader;

        protected bool _forwardInit;
        protected bool _backwardInit;

        protected NetworkLoss(ComputeShader shader)
        {
            Shader = shader;
        }

        public virtual void Calculate(float[,] output, float[,] yTrue)
        {
            if (!_forwardInit)
            {
                SampleLosses = new float[yTrue.GetLength(0)];
                _columnSize = output.GetLength(0);
                _rowSize = output.GetLength(1);

                _forwardInit = true;
            }

            for (int i = 0; i < _columnSize; i++)
            {
                var result = 0.0f;
                for (int j = 0; j < _rowSize; j++)
                {
                    float error = yTrue[i, j] - output[i, j];
                    result += error * error;
                }

                SampleLosses[i] = result / _rowSize;
            }
        }

        public virtual void Backward(float[,] output, float[,] yTrue)
        {
            if (!_backwardInit)
            {
                DInputs = new float[output.GetLength(0), output.GetLength(1)];
                _columnSize = output.GetLength(0);
                _rowSize = output.GetLength(1);

                _backwardInit = true;
            }

            for (int i = 0; i < _columnSize; i++)
            {
                for (int j = 0; j < _rowSize; j++)
                {
                    DInputs[i, j] = -2.0f * (yTrue[i, j] - output[i, j]) / _rowSize / _columnSize;
                }
            }
        }

        public virtual void Dispose()
        {
        }
    }

    public class NoLoss : NetworkLoss
    {
        public NoLoss(ComputeShader shader) : base(shader)
        {
        }

        public override void Calculate(float[,] output, float[,] yTrue)
        {
        }

        public override void Backward(float[,] output, float[,] yTrue)
        {
            DInputs = yTrue;
        }
    }

    public class MeanSquaredError : NetworkLoss
    {
        public MeanSquaredError(ComputeShader shader) : base(shader)
        {
        }
    }

    public class MeanSquaredErrorPrioritized : NetworkLoss
    {
        private float[] _sampleWeights;

        public MeanSquaredErrorPrioritized(ComputeShader shader) : base(shader)
        {
        }

        public void SetLossExternalParameters(float[] parameters)
        {
            
        }

        public override void Backward(float[,] output, float[,] yTrue)
        {
            if (!_backwardInit)
            {
                DInputs = new float[output.GetLength(0), output.GetLength(1)];
                _columnSize = output.GetLength(0);
                _rowSize = output.GetLength(1);

                _backwardInit = true;
            }

            for (int i = 0; i < _columnSize; i++)
            {
                float sampleWeight = _sampleWeights[i];
                for (int j = 0; j < _rowSize; j++)
                {
                    DInputs[i, j] = -2.0f * (yTrue[i, j] - output[i, j]) * sampleWeight / _rowSize / _columnSize;
                }
            }
        }
    }

    public class CategoricalCrossEntropy : NetworkLoss
    {
        private readonly float _minimum;
        private readonly float _maximum;
        
        public CategoricalCrossEntropy(ComputeShader shader) : base(shader)
        {
            _minimum = 1e-7f;
            _maximum = 1f - _minimum;
        }
        
        public override void Calculate(float[,] output, float[,] yTrue)
        {
            if (!_forwardInit)
            {
                SampleLosses = new float[yTrue.GetLength(0)];
                _columnSize = output.GetLength(0);
                _rowSize = output.GetLength(1);

                _forwardInit = true;
            }

            for (int i = 0; i < _columnSize; i++)
            {
                var result = 0.0f;
                for (int j = 0; j < _rowSize; j++)
                {
                    var outputValue = output[i, j];
                    if (outputValue < _minimum)
                    {
                        outputValue = _minimum;
                    }
                    else if (outputValue > _maximum)
                    {
                        outputValue = _maximum;
                    }
                    
                    result += yTrue[i, j] * Mathf.Log(outputValue);
                }

                SampleLosses[i] = result / _rowSize * -1f;
            }
        }
        public override void Backward(float[,] output, float[,] yTrue)
        {
            DInputs = yTrue;
        }
    }
}

/*
// Job class just for reference
       private TestJobsFor _testJob;
        private NativeArray<float> _yTrueJob;
        private NativeArray<float> _outputJob;
        private NativeArray<float> _sampleLossesJob;

        [BurstCompile]
        private struct TestJobsFor : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float> YTrue;
            [ReadOnly] public NativeArray<float> Output;
            [WriteOnly] public NativeArray<float> SampleLossesJob;

            public int ColumnSize;
            public int RowSize;

            public void Execute(int index)
            {
                SampleLossesJob[index] = -2.0f * (YTrue[index] - Output[index]) / RowSize / ColumnSize;
            }
        }
*/

/*

// Old class for reference, uses compute shader, fully functional
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
*/