using System.Collections.Generic;

namespace UMLNeuralNetwork
{
    public class HiddenNode : AbstractNode
    {
        private double _val;
        
        public HiddenNode(double val = 0.0)
        {
            _val = val;
        }
        public override double Val { get { return _val; } set { _val = value; } }
        public override List<Connector> OutputConnectors { get; set; }
        public override List<Connector> InputConnectors { get; set; }
    }
}
