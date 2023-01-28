namespace NN.CPU_Single
{
    public abstract class BaseLayer 
    {
        protected float[,] Inputs;
        public float[,] Output;
        public float[,] DInputs;

        public abstract void Forward(float[,] inputs);

        public abstract void Backward(float[,] dValues);
    }
}
