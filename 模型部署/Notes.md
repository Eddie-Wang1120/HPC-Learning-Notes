1存储层次
cache_read cache_write (shared memory)
set_scope              (计算结果存储层次 tensor最优存储位置)
storage_align          (factor为单位 offset为偏置对齐 避免bank conflict)
compute_at             (stage附着于指定iter方向并行计算)
compute_inline         (内联计算独立操作(computation fusion -> reduce stage))

2常见循环优化
fuse                   (内外循环融合为一层)
split                  (fuse相反 拆分循环为内外循环 形成更小的计算任务 与unroll不同)
reorder                (重置循环内外顺序 (M, N, K K放最外围))
tile                   (拆分循环为内外循环 计算任务更小)
unroll                 (循环展开 在一次iter中计算多个任务 减少iter次数)

3多线程并行优化
vectorize              (将循环迭代替换为ramp 当数据size为常量且为2的幂次时生效 用SIMD)
bind                   (iter绑定至blockidx 和 threadidx)
parallel               (for换为parallel 在cpu并行)

4其他schedule
pragma                 (unroll vectorize..)
prefectch              (前一个iter读数据 后一个iter计算)
tensorize              (将计算作为整体 调用内置intrinsic)
rfactor                (axis上做reduction)
set_store_predicate    (放置store条件)
create_group           (创建虚拟stage)

