using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace UMLNeuralNetwork
{
    public abstract class AbstractNode
    {
        public abstract double Val { get; set; }
        public abstract List<Connector> InputConnectors { get; set; }
        public abstract List<Connector> OutputConnectors { get; set; }
    }
}
