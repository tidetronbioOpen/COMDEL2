library(circlize)
library(RColorBrewer)
library(ComplexHeatmap)
library(gridBase)

args <- commandArgs(trailingOnly = T)
in_file = args[1] # "result/Combined_predict.csv"
out_file = args[2] # "predseq_circular_heatmap.png"

# 读取数据
data <- read.csv(in_file, header=TRUE, row.names= 1,sep=',')
data_matrix <- as.matrix(data)
head(data)
# 移除非数值列
#data_matrix <- as.matrix(data[, -c(1)])  #移除seq列
# 逆转行顺序以匹配顺时针方向
#rownames(data_matrix) <- rev(rownames(data_matrix))
#data_matrix <- data_matrix[nrow(data_matrix):1, ]
head(data_matrix)
# 获取数据矩阵的行数
n_rows <- nrow(data_matrix)
# 自定义颜色梯度
color_mapping <- colorRamp2(c(0, 0.5, 1), c("red", "black", "green"))

# 设置PNG图像设备
png(out_file, width = 800, height = 800)

#调整圆环首尾间距，开口朝上 
circos.par(start.degree = 90, gap.after = c(20))
# 绘制热图
circos.heatmap(data_matrix,col=color_mapping,
               dend.side="inside", #聚类放在环形内测,控制行聚类树的方向
               track.height = 0.6, #轨道的高度，数值越大圆环越粗
               bg.border = "black",#背景边缘颜色
               cluster=FALSE) #cluster=TRUE为对行聚类，cluster=FALSE则不显示聚类

# 添加自定义轴线
circos.axis(h = "top", labels.cex = 1.5, major.at = c(1, seq(2, n_rows, by=1)), 
            minor.ticks = 0, major.tick.length = 0.2, 
            major.tick.percentage = TRUE)

# 创建一个颜色条图例
lg = Legend(title = "Exp", col_fun = color_mapping,
            title_gp = gpar(fontsize = 12, fontface = "bold"),  # 调整标题字体大小和样式
            labels_gp = gpar(fontsize = 12),                   # 调整标签字体大小
            direction = "vertical",
            legend_width = unit(25, "cm"))  # 调整图例宽度
grid.draw(lg)
# 添加列名
circos.track(track.index = get.current.track.index(), panel.fun = function(x, y) {
  cn = colnames(data_matrix)
  n = length(cn)
  for (i in n:1) {  # 注意这里进行了逆序处理
    circos.text(CELL_META$cell.xlim[1], y = n - i + 1, labels = cn[i], cex = 1, adj = c(1, 1.5), facing = "inside", niceFacing = TRUE)
  }
}, bg.border = NA)

circos.clear()
# 关闭PDF设备
dev.off()
