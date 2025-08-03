#include "training.h"
#include <iomanip>
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

/**
 * @brief 对模型的所有梯度进行全局L2范数裁剪。
 *
 * @param params_and_grads 一个包含模型参数和梯度的容器。
 * @param max_norm 裁剪阈值。
 */
void clip_gradients(ParamsAndGrads &params_and_grads, float max_norm) {
  if (max_norm <= 0.0f) {
    return;
  }

  auto calculate_norm = [](dense::Tensor &grad) -> double {
    double norm = 0.0f;
    auto data = reinterpret_cast<const float *>(grad.data());
    for (size_t i = 0; i < grad.numel(); ++i) {
      norm += static_cast<double>(data[i] * data[i]);
    }
    return norm;
  };

  auto scale_grad = [](dense::Tensor &grad, double clip_coeff) {
    float *data = reinterpret_cast<float *>(grad.data());
    for (size_t i = 0; i < grad.numel(); ++i) {
      data[i] *= clip_coeff;
    }
  };

  double total_norm_sq = 0.0;

  for (auto &grad_group : params_and_grads.grads) {
    dense::Tensor &w = std::get<1>(grad_group);
    dense::Tensor &b = std::get<2>(grad_group);
    if (w.is_defined()) {
      total_norm_sq += calculate_norm(w);
    }
    if (b.is_defined()) {
      total_norm_sq += calculate_norm(b);
    }
  }
  double total_norm = std::sqrt(total_norm_sq);

  std::cout << "grad_norm =" << total_norm << std::endl;

  if (total_norm > max_norm) {
    double clip_coeff = max_norm / (total_norm + 1e-6);

    for (auto &grad_group : params_and_grads.grads) {
      dense::Tensor &w = std::get<1>(grad_group);
      dense::Tensor &b = std::get<2>(grad_group);
      if (w.is_defined()) {
        scale_grad(w, clip_coeff);
      }
      if (b.is_defined()) {
        scale_grad(b, clip_coeff);
      }
    }
  }
}

void WriteLog(const std::string &dir, const std::string &str) {
  std::string filename = dir + "/train.log";
  FILE *fp = fopen(filename.c_str(), "a");
  fprintf(fp, "%s\n", str.c_str());
  fclose(fp);
}
} // namespace

std::map<std::string, std::vector<double>>
train(GPT *model, const std::string &model_dir, DataLoader *train_data_loader,
      DataLoader *test_data_loader, LogSoftmaxCrossEntropyLoss *loss_func,
      AdamW *optimizer, CosineAnnealingWarmRestarts *scheduler,
      const TrainingArguments &args) {
  // 1. 初始化
  std::map<std::string, std::vector<double>> history;
  double best_val_loss = std::numeric_limits<double>::max();
  int patience_counter = 0;

  model->enable_training(true); // 确保模型处于训练模式

  double total_train_batches = static_cast<double>(train_data_loader->size());

  // 2. 主训练循环 (按 Epoch)
  for (int epoch = 0; epoch < args.epochs; ++epoch) {
    // --- 训练阶段 ---
    double running_loss = 0.0;
    int batch_count = 0;
    model->clear_grads();
    double accumulated_loss = 0.0;

    // 遍历训练数据加载器中的所有批次
    for (auto &batch : *train_data_loader) {
      auto X = batch.first;
      auto target = batch.second;

      auto logits = model->forward(X);
      auto loss = loss_func->forward(logits, target);
      auto grad_loss = loss_func->backward();

      if (args.accumulation_steps > 1) {
        // 在进行反向传播之前，将梯度除以累积步数
        // 这样可以确保在累积梯度后，更新的幅度与一个大批次更新的幅度大致相同
        auto N = grad_loss.numel();
        auto ptr = reinterpret_cast<float *>(grad_loss.data());
        for (size_t i = 0; i < N; ++i) {
          ptr[i] = ptr[i] / args.accumulation_steps;
        }
      }
      model->backward(grad_loss);

      accumulated_loss += loss;
      batch_count++;

      double current_epoch_progress =
          static_cast<double>(epoch) +
          (static_cast<double>(batch_count) / total_train_batches);

      scheduler->step(current_epoch_progress);
      auto lr = scheduler->get_lr();
      optimizer->set_learning_rate(lr);

      {
        char buf[1024] = {0};
        sprintf(buf, "Epoch:%d, Batch:%d,Loss: %.8f,LR: %.8f", epoch + 1,
                batch_count, loss, lr);
        WriteLog(model_dir, buf);
      }

      std::cout << "Epoch:" << epoch + 1 << ",Batch:" << batch_count
                << ", Training Loss: " << loss << ", LR: " << std::fixed
                << std::setprecision(8) << lr << std::endl;

      if (batch_count % args.accumulation_steps == 0) {

        ParamsAndGrads params_and_grads;
        model->get_params_and_grads(params_and_grads);

        clip_gradients(params_and_grads, args.max_grad_norm);

        // 使用优化器更新参数
        optimizer->update(params_and_grads);
        model->clear_grads();

        running_loss += accumulated_loss / args.accumulation_steps;

        {
          char buf[1024] = {0};
          sprintf(buf, "Epoch:%d, Batch:%d,Loss (accumulated): %.8f", epoch + 1,
                  batch_count, accumulated_loss / args.accumulation_steps);
          WriteLog(model_dir, buf);
        }

        std::cout << "Epoch:" << epoch + 1 << ",Batch:" << batch_count
                  << ", Training Loss (accumulated): "
                  << accumulated_loss / args.accumulation_steps << std::endl;

        // 重置累积损失
        accumulated_loss = 0.0;
      }
    }

    // 处理最后一个不完整的梯度累积批次
    if (batch_count % args.accumulation_steps != 0) {
      ParamsAndGrads params_and_grads;
      model->get_params_and_grads(params_and_grads);
      clip_gradients(params_and_grads, args.max_grad_norm);
      optimizer->update(params_and_grads);
      model->clear_grads();
      running_loss +=
          accumulated_loss / (batch_count % args.accumulation_steps);

      {
        char buf[1024] = {0};
        sprintf(buf, "Epoch:%d, Batch:%d,Loss (accumulated): %.8f", epoch + 1,
                batch_count, accumulated_loss / args.accumulation_steps);
        WriteLog(model_dir, buf);
      }

      std::cout << "Epoch:" << epoch + 1 << ",Batch:" << batch_count
                << ", Training Loss (accumulated): "
                << accumulated_loss / args.accumulation_steps << std::endl;
    }

    double avg_train_loss = running_loss / batch_count;
    history["train_loss"].push_back(avg_train_loss);
    std::cout << "Epoch " << epoch + 1
              << " Average Train Loss: " << avg_train_loss << std::endl;

    // --- 评估阶段 ---
    if ((epoch + 1) % args.eval_interval == 0) {
      model->enable_training(false); // 禁用训练模式
      float val_loss = evaluate(model, test_data_loader, loss_func);

      model->enable_training(true); // 恢复训练模式
      history["val_loss"].push_back(val_loss);
      std::cout << "Epoch " << epoch + 1 << " Validation Loss: " << val_loss
                << std::endl;

      // --- 早停与保存最佳模型逻辑 ---
      if (val_loss < best_val_loss - args.min_delta) {
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
                  << patience_counter << "/" << args.patience << std::endl;
      }

      if (patience_counter >= args.patience) {
        std::cout << "Early stopping triggered after " << epoch + 1
                  << " epochs." << std::endl;
        break; // 退出训练循环
      }
    }
  }
  std::cout << "Training finished." << std::endl;
  return history;
}