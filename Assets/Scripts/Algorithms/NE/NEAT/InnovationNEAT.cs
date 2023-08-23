using System.Collections.Generic;

namespace Algorithms.NE.NEAT
{
    public class InnovationNEAT
    {
        private readonly List<Innovation> _innovations;
        private readonly int _inputNumber;
        private readonly int _outputNumber;
        private int _neuronIdCount;
        private int _innovationIdCount;

        public InnovationNEAT(int inputNumber, int outputNumber)
        {
            _inputNumber = inputNumber;
            _outputNumber = outputNumber;

            _innovations = new List<Innovation>(inputNumber + outputNumber + inputNumber * outputNumber);
            var neuronId = 1;
            for (int i = 0; i < inputNumber; i++)
            {
                _innovations.Add(new Innovation(-1, -1, -1, neuronId++, NeuronType.Input));
            }

            for (int i = 0; i < outputNumber; i++)
            {
                _innovations.Add(new Innovation(-1, -1, -1, neuronId++, NeuronType.Output));
            }

            _neuronIdCount = neuronId;
            var innovationNumber = 1;
            for (int i = 0; i < inputNumber; i++)
            {
                var inputNeuronId = _innovations[i].NeuronId;
                for (int j = 0; j < outputNumber; j++)
                {
                    _innovations.Add(new Innovation(innovationNumber++, inputNeuronId, _innovations[j].NeuronId, -1,
                        NeuronType.None));
                }
            }

            _innovationIdCount = innovationNumber;
        }

        public int CheckInnovation(int neuron1Id, int neuron2Id, NeuronType neuronType)
        {
            var searchStart = _inputNumber + _outputNumber + _inputNumber * _outputNumber - _innovations.Count;
            for (int i = 0; i < searchStart; i++)
            {
                var innovation = _innovations[i];

                if (innovation.NeuronType == neuronType && innovation.NeuronIn == neuron1Id &&
                    innovation.NeuronOut == neuron2Id)
                    return neuronType == NeuronType.None ? innovation.InnovationId : innovation.NeuronId;
            }

            return -1;
        }

        public int CreateNewInnovation(int neuron1Id, int neuron2Id, NeuronType neuronType)
        {
            int id;
            if (neuronType == NeuronType.None)
            {
                id = _innovationIdCount++;
                _innovations.Add(new Innovation(id, neuron1Id, neuron2Id, -1, neuronType));
            }
            else
            {
                id = _neuronIdCount++;
                _innovations.Add(new Innovation(-1, neuron1Id, neuron2Id, id, neuronType));
            }

            return id;
        }
    }

    public struct Innovation
    {
        public readonly int InnovationId;
        public readonly int NeuronIn;
        public readonly int NeuronOut;
        public readonly int NeuronId;
        public readonly NeuronType NeuronType;

        public Innovation(int innovationId, int neuronIn, int neuronOut, int neuronId, NeuronType neuronType)
        {
            InnovationId = innovationId;
            NeuronIn = neuronIn;
            NeuronOut = neuronOut;
            NeuronId = neuronId;
            NeuronType = neuronType;
        }
    }
}