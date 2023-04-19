namespace Algorithms
{
    public class SumTree
    {
        private readonly float[] _tree;
        private readonly int _size;
        private readonly int _treeSize;

        public SumTree(int size)
        {
            _size = size;
            _treeSize = 2 * size - 1;
            _tree = new float[_treeSize];
        }

        public float Total()
        {
            return _tree[0];
        }

        public float Get(int index)
        {
            return _tree[index + _size - 1];
        }

        public void UpdateValue(int index, float value)
        {
            var treeIndex = index + _size - 1;

            var change = value - _tree[treeIndex];
            _tree[treeIndex] = value;

            Propagate(treeIndex, change);
        }

        private void Propagate(int treeIndex, float change)
        {
            while (true)
            {
                var parent = (treeIndex - 1) / 2;
                _tree[parent] += change;

                if (parent != 0)
                {
                    treeIndex = parent;
                    continue;
                }

                break;
            }
        }

        //TODO: can also return the value
        public int Sample(float value, out float prio)
        {
            var treeIndex = Retrieve(0, value);
            prio = _tree[treeIndex];
            return treeIndex - _size + 1;
        }

        private int Retrieve(int treeIndex, float value)
        {
            while (true)
            {
                var left = 2 * treeIndex + 1;
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