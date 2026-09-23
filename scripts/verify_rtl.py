"""Verify deterministic RTL traces with Yosys sequential SAT (no PDK required)."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from squant.hardware import reconstruct_numerators, generator_trace


def literal(value, width=48):
    return ("-" if value < 0 else "")+f"{width}'sd{abs(int(value))}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yosys", default=shutil.which("yosys") or shutil.which("yowasp-yosys"))
    parser.add_argument("--output", default="results/rtl")
    args = parser.parse_args()
    if not args.yosys:
        parser.error("Install Yosys or pip install yowasp-yosys, or supply --yosys")
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cases = [(1, [-3, 7, 0]), (173, [127, -128, 13, 4])]
    code = ["module verify_top(input clk, rst, output passed, output reg failed);",
            "reg [7:0] cycle;",
            "always @(posedge clk) if (rst) cycle<=0; else cycle<=cycle+1;",
            "wire go = cycle==1; wire accept = cycle%3 != 0;"]
    checks = []
    for index, (seed, coeff) in enumerate(cases):
        expected = reconstruct_numerators(seed, coeff, 8, 8, 0xB8)
        packed = sum((q & 255) << (8*k) for k, q in enumerate(coeff))
        code += [f"wire valid{index}, done{index}; wire [2:0] ix{index}; wire signed [47:0] num{index};",
                 f"reg seen{index}; reg [4:0] count{index}; reg signed [47:0] expected{index};",
                 f"weight_generator #(.S(8),.B(8),.MAX_K(4)) gen{index} (.clk(clk), .rst_n(!rst), .start(go), .ready(),",
                 f".seed(8'd{seed}), .basis_count(3'd{len(coeff)}), .coefficients(32'h{packed:08x}),",
                 f".out_valid(valid{index}), .out_ready(accept), .out_numerator(num{index}), .out_index(ix{index}), .done(done{index}));",
                 f"always @* begin expected{index}=0; case(ix{index})"]
        code += [f"3'd{i}: expected{index}={literal(value)};" for i, value in enumerate(expected)]
        code += ["endcase end",
                 f"always @(posedge clk) if(rst) begin seen{index}<=0; count{index}<=0; end else begin",
                 f"if(done{index}) seen{index}<=1; if(valid{index} && accept) count{index}<=count{index}+1; end"]
        checks.append(f"(valid{index} && accept && (num{index} != expected{index} || ix{index} != count{index}))")
    a = np.array([[1,-2,3],[4,5,-6]])
    b = np.array([[7,8],[-9,10],[11,-12]])
    code += ["reg [31:0] aa,bb; reg [1:0] av,bv; wire [191:0] sums; reg array_checked;",
             "systolic_array array0(.clk(clk),.rst_n(!rst),.clear(1'b0),.a_rows(aa),.b_cols(bb),.a_valid(av),.b_valid(bv),.result(sums));",
             "always @* begin aa=0; bb=0; av=0; bv=0; case(cycle)"]
    for t in range(4):
        ar, bc, valid = 0, 0, 0
        for i in range(2):
            k = t-i
            if 0 <= k < 3:
                ar |= (int(a[i,k]) & 65535) << (16*i)
                bc |= (int(b[k,i]) & 65535) << (16*i)
                valid |= 1 << i
        code.append(f"8'd{t+2}: begin aa=32'h{ar:08x}; bb=32'h{bc:08x}; av=2'd{valid}; bv=2'd{valid}; end")
    code += ["endcase end", "always @(posedge clk) if(rst) array_checked<=0; else if(cycle==10) array_checked<=1;"]
    product = a@b
    array_checks = [f"$signed(sums[{i*48} +: 48]) != {literal(value)}" for i, value in enumerate(product.ravel())]
    checks.append("(cycle==10 && ("+" || ".join(array_checks)+"))")
    code += ["always @(posedge clk) if(rst) failed<=0; else if ("+" || ".join(checks)+") failed<=1;",
             "assign passed = !failed && array_checked && seen0 && seen1 && count0==8 && count1==8;", "endmodule"]
    tb = output/"verify_top.sv"
    tb.write_text("\n".join(code)+"\n")
    yosys_script = output/"verify.ys"
    yosys_script.write_text(f'read_verilog -sv hardware/weight_generator.sv hardware/systolic_array.sv "{tb}"\n'
        'prep -top verify_top -flatten\nmemory_map\nopt\ncheck\n'
        'sat -seq 65 -set-init-zero -set rst 0 -set-at 1 rst 1 -prove-skip 64 -prove passed 1 -verify\n')
    env = os.environ.copy()
    env.setdefault("YOWASP_CACHE_DIR", str(root/".cache/yowasp"))
    result = subprocess.run([args.yosys, "-Q", "-T", "-s", str(yosys_script)], capture_output=True, text=True, env=env)
    (output/"verification.log").write_text(result.stdout+result.stderr)
    if result.returncode or "SUCCESS" not in result.stdout:
        raise RuntimeError(f"RTL check failed; see {output/'verification.log'}")
    report = {"status": "passed", "method": "Yosys bounded sequential SAT, 65 clock steps, deterministic fixtures",
              "checked": ["two seeded integer reconstruction streams", "negative and extreme int8 coefficients",
                          "backpressure and output order", "completion", "2x2 systolic signed matrix multiplication"],
              "ppa_evaluated": False}
    (output/"report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
