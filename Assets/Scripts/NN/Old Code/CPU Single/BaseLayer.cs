namespace NN.CPU_Single
{
    public abstract class BaseLayer 
    {
        public float[,] Inputs;
        public float[,] Output;
        public float[,] DInputs;

        public abstract void Forward(float[,] inputs);

        public abstract void Backward(float[,] dValues);
    }
}
