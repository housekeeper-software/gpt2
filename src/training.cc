#include "training.h"
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
float evaluate(GPT *model, DataLoader *test_data_loader,
               LogSoftmaxCrossEntropyLoss *loss_func) {
  double total_loss = 0.0;
  int num_batches = 0;
  for (const auto &batch : *test_data_loader) {
    auto X = batch.first;
    auto target = batch.second;
    auto logits = model->forward(X);
    auto loss = loss_func->forward(logits, target);
    total_loss += loss;
    num_batches++;
  }
  if (num_batches < 1)
    return 0.0f;
  return static_cast<float>(total_loss / num_batches);
}
} // namespace

std::map<std::string, std::vector<double>>
train(GPT *model, DataLoader *X_train_data_loader,
      DataLoader *X_test_data_loader, LogSoftmaxCrossEntropyLoss *loss_func,
      AdamW *optimizer, int epochs, const std::string &model_dir,
      int accumulation_steps, int T_0, int T_mult, double eta_min,
      int eval_interval, int patience, double min_delta) {
  // 1. 初始化
  std::map<std::string, std::vector<double>> history;
  double best_val_loss = std::numeric_limits<double>::max();
  int patience_counter = 0;

  // 学习率调度器状态
  double initial_lr = optimizer->get_learning_rate();
  int current_epoch_in_cycle = 0;
  int T_current = T_0;

  model->enable_training(true); // 确保模型处于训练模式

  // 2. 主训练循环 (按 Epoch)
  for (int epoch = 0; epoch < epochs; ++epoch) {
    // --- 学习率更新
    double phase = std::_Pi_val *
                   (static_cast<double>(current_epoch_in_cycle) / T_current);
    double lr =
        eta_min + 0.5 * (initial_lr - eta_min) * (1.0 + std::cos(phase));
    optimizer->set_learning_rate(lr);

    std::cout << "Epoch " << epoch + 1 << "/" << epochs
              << ", LR: " << optimizer->get_learning_rate() << std::endl;

    // --- 训练阶段 ---
    double running_loss = 0.0;
    int batch_count = 0;
    model->clear_grads();
    double accumulated_loss = 0.0;

    // 遍历训练数据加载器中的所有批次
    for (auto &batch : *X_train_data_loader) {
      auto X = batch.first;
      auto target = batch.second;

      auto logits = model->forward(X);
      auto loss = loss_func->forward(logits, target);
      auto grad_loss = loss_func->backward();

      if (accumulation_steps > 1) {
        // 在进行反向传播之前，将梯度除以累积步数
        // 这样可以确保在累积梯度后，更新的幅度与一个大批次更新的幅度大致相同
        auto N = grad_loss.numel();
        auto ptr = reinterpret_cast<float *>(grad_loss.data());
        for (size_t i = 0; i < N; ++i) {
          ptr[i] = ptr[i] / accumulation_steps;
        }
      }
      model->backward(grad_loss);

      accumulated_loss += loss;
      batch_count++;

      std::cout << "Epoch:" << epoch + 1 << ",Batch:" << batch_count
                << ", Training Loss: " << loss << std::endl;

      if (batch_count % accumulation_steps == 0) {
        ParamsAndGrads params_and_grads;
        model->get_params_and_grads(params_and_grads);
        // 使用优化器更新参数
        optimizer->update(params_and_grads);
        model->clear_grads();

        running_loss += accumulated_loss / accumulation_steps;

        std::cout << "Epoch:" << epoch + 1 << ",Batch:" << batch_count
                  << ", Training Loss (accumulated): "
                  << accumulated_loss / accumulation_steps << std::endl;

        // 重置累积损失
        accumulated_loss = 0.0;
      }
    }

    // 处理最后一个不完整的梯度累积批次
    if (batch_count % accumulation_steps != 0) {
      ParamsAndGrads params_and_grads;
      model->get_params_and_grads(params_and_grads);
      optimizer->update(params_and_grads);
      model->clear_grads();
      running_loss += accumulated_loss / (batch_count % accumulation_steps);
    }

    double avg_train_loss = running_loss / batch_count;
    history["train_loss"].push_back(avg_train_loss);
    std::cout << "Epoch " << epoch + 1
              << " Average Train Loss: " << avg_train_loss << std::endl;

    // --- 评估阶段 ---
    if ((epoch + 1) % eval_interval == 0) {
      model->enable_training(false); // 禁用训练模式
      float val_loss = evaluate(model, X_test_data_loader, loss_func);

      model->enable_training(true); // 恢复训练模式
      history["val_loss"].push_back(val_loss);
      std::cout << "Epoch " << epoch + 1 << " Validation Loss: " << val_loss
                << std::endl;

      // --- 早停与保存最佳模型逻辑 ---
      if (val_loss < best_val_loss - min_delta) {
        best_val_loss = val_loss;
        patience_counter = 0;
        char buffer[1024];
        snprintf(buffer, sizeof(buffer), "%s/best_%d_%.4f.safetensors",
                 model_dir.c_str(), epoch + 1, val_loss);
        std::cout << "Validation loss improved. Saving model to " << buffer
                  << std::endl;
        model->save(buffer);
      } else {
        patience_counter++;
        std::cout << "Validation loss did not improve. Patience: "
                  << patience_counter << "/" << patience << std::endl;
      }

      if (patience_counter >= patience) {
        std::cout << "Early stopping triggered after " << epoch + 1
                  << " epochs." << std::endl;
        break; // 退出训练循环
      }
    }

    // --- 更新学习率调度器状态 ---
    current_epoch_in_cycle++;
    if (current_epoch_in_cycle >= T_current) {
      current_epoch_in_cycle = 0; // 重置周期内的 epoch 计数
      T_current *= T_mult;        // 增加下一个周期的长度
      initial_lr =
          optimizer->get_learning_rate(); // 将当前学习率作为下一个周期的起始点
      std::cout << "LR scheduler restart. Next cycle T=" << T_current
                << std::endl;
    }
  }
  std::cout << "Training finished." << std::endl;
  return history;
}