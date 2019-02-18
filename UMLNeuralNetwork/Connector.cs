using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace UMLNeuralNetwork
{
    public class Connector
    {
        public Connector(double weight = .05, double weightDelta = 0.011)
        {
            Weight = weight;
            WeightDelta = weightDelta;
        }
        public double Weight { get; set; }
        public double WeightDelta { get; set; }
        public AbstractNode FromNode { get; set; }
        public AbstractNode ToNode { get; set; }
    }
}
