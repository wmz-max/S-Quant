module systolic_array #(
    parameter integer N=2,
    parameter integer DATA_W=16,
    parameter integer ACC_W=48
) (
    input logic clk, rst_n, clear,
    input logic [N*DATA_W-1:0] a_rows,
    input logic [N*DATA_W-1:0] b_cols,
    input logic [N-1:0] a_valid,
    input logic [N-1:0] b_valid,
    output wire [N*N*ACC_W-1:0] result
);
    wire signed [DATA_W-1:0] a [0:N-1][0:N];
    wire signed [DATA_W-1:0] b [0:N][0:N-1];
    wire av [0:N-1][0:N];
    wire bv [0:N][0:N-1];
    genvar i,j;
    generate
        for (i=0; i<N; i=i+1) begin: boundaries
            assign a[i][0] = a_rows[i*DATA_W +: DATA_W];
            assign av[i][0] = a_valid[i];
            assign b[0][i] = b_cols[i*DATA_W +: DATA_W];
            assign bv[0][i] = b_valid[i];
        end
        for (i=0; i<N; i=i+1) begin: rows
            for (j=0; j<N; j=j+1) begin: cols
                logic signed [DATA_W-1:0] a_reg, b_reg;
                logic av_reg, bv_reg;
                logic signed [ACC_W-1:0] sum;
                wire signed [2*DATA_W-1:0] product = a[i][j]*b[i][j];
                assign a[i][j+1] = a_reg;
                assign b[i+1][j] = b_reg;
                assign av[i][j+1] = av_reg;
                assign bv[i+1][j] = bv_reg;
                assign result[(i*N+j)*ACC_W +: ACC_W] = sum;
                always_ff @(posedge clk) begin
                    if (!rst_n || clear) begin
                        a_reg <= 0; b_reg <= 0;
                        av_reg <= 0; bv_reg <= 0;
                        sum <= 0;
                    end else begin
                        a_reg <= a[i][j]; b_reg <= b[i][j];
                        av_reg <= av[i][j]; bv_reg <= bv[i][j];
                        if (av[i][j] && bv[i][j]) sum <= sum+product;
                    end
                end
            end
        end
    endgenerate
endmodule
