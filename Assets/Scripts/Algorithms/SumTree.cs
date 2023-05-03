using System.Collections.Generic;

namespace Algorithms
{
    public class SumTree
    {
        private readonly float[] _tree;
        private readonly int _size;
        private readonly int _treeSize;

        //private readonly float[] _maxTree;

        public SumTree(int size, IReadOnlyList<float> array = null)
        {
            _size = size;
            _treeSize = 2 * size;
            _tree = new float[_treeSize];

            //_maxTree = new float[size];

            if (array == null) return;

            for (int i = 0; i < size; i++)
            {
                _tree[i + size] = array[i];
            }

            for (int i = size - 1; i > 0; i--)
            {
                var left = i * 2;
                _tree[i] = _tree[left] + _tree[left + 1];

                // var arr = left < size ? _maxTree : _tree;
                // _maxTree[i] = arr[left] > arr[left + 1] ? arr[left] : arr[left + 1];
            }
        }

        public float Total()
        {
            return _tree[1];
        }
        
        public float MaxValue()
        {
            return 0.0f;
            //return _maxTree[1];
        }

        public float Get(int index)
        {
            return _tree[index + _size];
        }

        public void UpdateValue(int index, float value)
        {
            index += _size;

            var change = value - _tree[index];
            _tree[index] = value;

            index /= 2;
            _tree[index] += change;
            
            // var left = index * 2;
            // _maxTree[index] = _tree[left] > _tree[left + 1] ? _tree[left] : _tree[left + 1];
            
            while (index > 1)
            {
                index /= 2;
                _tree[index] += change;
                
                // left = index * 2;
                // var leftValue = _maxTree[left];
                // var rightValue = _maxTree[left + 1];
                // _maxTree[index] = leftValue > rightValue ? leftValue : rightValue;
            }
        }

        public int Sample(float randomValue, out float treeValue)
        {
            var treeIndex = Retrieve(1, randomValue);
            treeValue = _tree[treeIndex];
            return treeIndex - _size;
        }

        private int Retrieve(int treeIndex, float value)
        {
            while (true)
            {
                var left = 2 * treeIndex;
                var right = left + 1;

                if (left >= _treeSize)
                {
                    return treeIndex;
                }

                var leftValue = _tree[left];
                if (value <= leftValue)
                {
                    treeIndex = left;
                    continue;
                }

                treeIndex = right;
                value -= leftValue;
            }
        }
    }
}