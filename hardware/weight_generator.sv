module weight_generator #(
    parameter integer S = 8,
    parameter integer B = 8,
    parameter integer MAX_K = 8,
    parameter integer ACC_W = 48,
    parameter logic [S-1:0] MASK = 8'hb8
) (
    input logic clk, rst_n,
    input logic start,
    output logic ready,
    input logic [S-1:0] seed,
    input logic [$clog2(MAX_K+1)-1:0] basis_count,
    input logic [MAX_K*8-1:0] coefficients,
    output logic out_valid,
    input logic out_ready,
    output logic signed [ACC_W-1:0] out_numerator,
    output logic [$clog2(B)-1:0] out_index,
    output logic done
);
    localparam integer IDLE = 0, COMPUTE = 1, EMIT = 2;
    integer mode, b_index, k_index, emit_index, k_latched;
    logic [S-1:0] lfsr;
    logic [MAX_K*8-1:0] coeff_latched;
    logic signed [ACC_W-1:0] accum [0:B-1];
    logic signed [S:0] centered;
    logic signed [7:0] coefficient;
    logic signed [S+8:0] product;
    integer i;

    always_comb begin
        ready = (mode == IDLE);
        out_valid = (mode == EMIT);
        out_index = emit_index;
        out_numerator = accum[emit_index];
        centered = $signed({1'b0, lfsr}) - (1 << (S-1));
        coefficient = $signed(coeff_latched[k_index*8 +: 8]);
        product = centered * coefficient;
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            mode <= IDLE;
            lfsr <= 1;
            b_index <= 0;
            k_index <= 0;
            emit_index <= 0;
            k_latched <= 0;
            coeff_latched <= 0;
            done <= 0;
            for (i=0; i<B; i=i+1) accum[i] <= 0;
        end else begin
            done <= 0;
            case (mode)
                IDLE: if (start && seed != 0 && basis_count > 0 && basis_count <= MAX_K) begin
                    lfsr <= seed;
                    k_latched <= basis_count;
                    coeff_latched <= coefficients;
                    b_index <= 0;
                    k_index <= 0;
                    emit_index <= 0;
                    for (i=0; i<B; i=i+1) accum[i] <= 0;
                    mode <= COMPUTE;
                end
                COMPUTE: begin
                    accum[b_index] <= accum[b_index] + product;
                    lfsr <= (lfsr >> 1) ^ (lfsr[0] ? MASK : {S{1'b0}});
                    if (b_index == B-1) begin
                        b_index <= 0;
                        if (k_index == k_latched-1) begin
                            mode <= EMIT;
                            emit_index <= 0;
                        end else k_index <= k_index+1;
                    end else b_index <= b_index+1;
                end
                EMIT: if (out_ready) begin
                    if (emit_index == B-1) begin
                        mode <= IDLE;
                        done <= 1;
                    end else emit_index <= emit_index+1;
                end
                default: mode <= IDLE;
            endcase
        end
    end
endmodule
