#include <torch/torch.h>
#include <xilinx_ocl_helper.h>
#include <event_timer.h>

#include <iostream>
#include <string>

// Xilinx OpenCL and XRT includes

struct Net : torch::nn::Module {
   Net(int64_t N, int64_t M)
      : linear(register_module("linear", torch::nn::Linear(N, M))) {
    another_bias = register_parameter("b", torch::randn(M));
  }
  torch::Tensor forward(torch::Tensor input) {
    return linear(input) + another_bias;
  }
  torch::nn::Linear linear;
  torch::Tensor another_bias;
};


// Inherit from Function
class LinearFunction : public torch::autograd::Function<LinearFunction> {
 public:
  // Note that both forward and backward are static functions

  // bias is an optional argument
  static torch::Tensor forward(
      torch::autograd::AutogradContext *ctx, 
      torch::Tensor input, 
      torch::Tensor weight, 
      torch::Tensor bias = torch::Tensor()) 
  {
    ctx->save_for_backward({input, weight, bias});
    // aten::mm(Tensor self, Tensor mat2) -> Tensor
    auto output = input.mm(weight.t());
    if (bias.defined()) {
      output += bias.unsqueeze(0).expand_as(output);
    }
    return output;
  }

  static torch::autograd::tensor_list backward(
    torch::autograd::AutogradContext *ctx, 
    torch::autograd::tensor_list grad_outputs) 
  {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto weight = saved[1];
    auto bias = saved[2];

    auto grad_output = grad_outputs[0];
    auto grad_input = grad_output.mm(weight);
    auto grad_weight = grad_output.t().mm(input);
    auto grad_bias = torch::Tensor();
    if (bias.defined()) {
      grad_bias = grad_output.sum(0);
    }

    return {grad_input, grad_weight, grad_bias};
  }
};


int main(){
    torch::Tensor tensor = torch::rand({2, 3});
    float *ptr = (float*)tensor.storage().data_ptr().get();

    std::cout << "-- Test 0: Get torch.Tensor Memory pointer -- " << std::endl;
    std::cout << "> Tensor Memory pointer" << std::endl;
    std::cout << ptr <<std::endl;
    std::cout << "> torch.Tensor" << std::endl;
    std::cout << tensor << std::endl;
    std::cout << std::endl;

    std::cout << "-- Test 1: Modify torch.Tensor element -- " << std::endl;
    *ptr = (float)0.0;
    std::cout << "> Tensor pointer elem" << std::endl;
    std::cout << *ptr <<std::endl;
    std::cout << "> torch.Tensor" << std::endl;
    std::cout << tensor << std::endl;
    std::cout << std::endl;

    std::cout << "-- Test 2: Get torch.Tensor layout -- " << std::endl;
    std::cout << "> Tensor pointer elem" << std::endl;
    std::cout << *ptr <<std::endl;
    std::cout << "> torch.Tensor" << std::endl;
    std::cout << tensor << std::endl;
    std::cout << std::endl;

    std::cout << "-- Test Done -- " << std::endl;
    std::cout << std::endl;
    //////////////////////////////////////////////////////
    std::cout << "====== Running \"Using custom autograd function in C++\" ======" << std::endl;
    {
      auto x = torch::randn({2, 3}).requires_grad_();
      auto weight = torch::randn({4, 3}).requires_grad_();
      auto y = LinearFunction::apply(x, weight);
      y.sum().backward();

      std::cout << x.grad() << std::endl;
      std::cout << weight.grad() << std::endl;
    }
    //////////////////////////////////////////////////////
    Net net(4, 5);
    for (const auto& p : net.parameters()) {
        //std::cout << p << std::endl;
    }
    //////////////////////////////////////////////////////
    EventTimer et;
    std::cout << "-- Example 0: Loading the FPGA Binary --" << std::endl
              << std::endl;

    // Initialize the runtime (including a command queue) and load the
    // FPGA image
    std::cout << "Loading alveo_examples.xclbin to program the Alveo board" << std::endl
              << std::endl;
    et.add("OpenCL Initialization");

    xilinx::example_utils::XilinxOclHelper xocl;
    xocl.initialize("alveo_examples.xclbin");

    cl::CommandQueue q = xocl.get_command_queue();
    cl::Kernel krnl    = xocl.get_kernel("vadd");
    et.finish();

    std::cout << std::endl
              << "FPGA programmed, example complete!" << std::endl
              << std::endl;

    std::cout << "-- Key execution times --" << std::endl;

    et.print();

    //cl::CommandQueue q = xocl.get_command_queue();
    //cl::Kernel krnl = xocl.get_kernel("VectorAdd_hw_emu");
};
