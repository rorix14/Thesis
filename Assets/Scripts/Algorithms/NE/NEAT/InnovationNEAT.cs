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
                _innovations.Add(new Innovation(neuronId++, -1, -1, NeuronType.Input));
            }

            for (int i = 0; i < outputNumber; i++)
            {
                _innovations.Add(new Innovation(neuronId++, -1, -1, NeuronType.Output));
            }

            _neuronIdCount = neuronId;
            var innovationNumber = 1;
            for (int i = 0; i < inputNumber; i++)
            {
                var inputNeuronId = _innovations[i].Id;
                for (int j = 0; j < outputNumber; j++)
                {
                    _innovations.Add(new Innovation(innovationNumber++, inputNeuronId, _innovations[j].Id,
                        NeuronType.None));
                }
            }

            _innovationIdCount = innovationNumber;
        }

        public int CheckInnovation(int neuron1Id, int neuron2Id, NeuronType neuronType)
        {
            var searchStart = _inputNumber + _outputNumber + _inputNumber * _outputNumber;
            for (int i = searchStart; i < _innovations.Count; i++)
            {
                var innovation = _innovations[i];

                if (innovation.NeuronType == neuronType && innovation.NeuronIn == neuron1Id &&
                    innovation.NeuronOut == neuron2Id)
                    return innovation.Id;
            }

            return -1;
        }

        public int CreateNewInnovation(int neuron1Id, int neuron2Id, NeuronType neuronType)
        {
            int id;
            if (neuronType == NeuronType.None)
            {
                id = _innovationIdCount++;
                _innovations.Add(new Innovation(id, neuron1Id, neuron2Id, neuronType));
            }
            else
            {
                id = _neuronIdCount++;
                _innovations.Add(new Innovation(id, neuron1Id, neuron2Id, neuronType));
            }

            return id;
        }
    }

    public struct Innovation
    {
        public readonly int Id;
        public readonly int NeuronIn;
        public readonly int NeuronOut;
        public readonly NeuronType NeuronType;

        public Innovation(int id, int neuronIn, int neuronOut, NeuronType neuronType)
        {
            Id = id;
            NeuronIn = neuronIn;
            NeuronOut = neuronOut;
            NeuronType = neuronType;
        }
    }
}