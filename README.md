# VideoEspresso
\[[Paper](https://arxiv.org/abs/2411.14794)] [[Test Set](https://huggingface.co/datasets/hshjerry0315/VideoEspresso-Test)]

**Announcement: Leaderboard Coming Soon!** 🚀

We are excited to announce that our **leaderboard** will be updated and filled shortly.  

## Leaderboard
| Model               | Params | Frames | Overall | Narrative Analysis | Event Dynamic | Preparation Steps | Causal Analysis | Theme Analysis | Contextual Analysis | Influence Analysis | Role Analysis | Interaction Analysis | Behavior Analysis | Emotion Analysis | Cooking Process | Traffic Analysis | Situation Analysis |
|----------------------|--------|--------|---------|---------------------|------------------------|------------------------------------------|-----------------|----------------|---------------------|--------------------|---------------|----------------------|------------------|------------------|----------------|------------------|--------------------|
| **LLaVA-Video**     | 72B    | 64     | **66.3%** | 68.4%              | **66.2%**             | **74.5%**                               | **62.7%**       | 62.3%         | **71.6%**           | 62.5%             | **63.5%**     | **67.7%**            | **63.2%**        | 60.0%           | 75.5%         | **76.7%**        | **74.0%**          |
| **LLaVA-OneVision** | 72B    | 64     | 63.2%  | **76.0%**          | 61.8%                 | 71.4%                                   | 57.5%          | 62.3%         | 68.8%              | 62.5%             | 55.6%        | 58.1%               | 56.1%           | **63.1%**       | **77.4%**     | 70.0%           | 74.0%             |
| **InternVL2.5**     | 38B    | 16     | 59.9%  | 65.8%              | 54.1%                 | 66.3%                                   | 57.3%          | 55.7%         | 63.3%              | 56.9%             | 54.0%        | 53.2%               | 63.2%           | 60.0%           | 73.6%         | 70.0%           | 72.0%             |
| **Gemini-1.5-pro**  | -      | 128    | 44.2%  | 55.7%              | 42.0%                 | 50.0%                                   | 41.3%          | 34.4%         | 53.2%              | 29.2%             | 39.7%        | 40.3%               | 38.6%           | 47.7%           | 58.5%         | 50.0%           | 54.0%             |
| **Qwen-Max**        | -      | 4      | 42.7%  | 44.3%              | 35.7%                 | 45.9%                                   | 39.7%          | 44.3%         | 54.1%              | 43.1%             | 47.6%        | 35.5%               | 45.6%           | 41.5%           | 49.1%         | 46.7%           | 46.0%             |
| **Gemini-1.5-flash**| -      | 128    | 39.8%  | 59.5%              | 45.2%                 | 38.8%                                   | 34.7%          | 32.8%         | 45.9%              | 30.6%             | 42.9%        | 43.6%               | 33.3%           | 38.5%           | 41.5%         | 36.7%           | 46.0%             |
| **LongVA**          | 7B     | 128    | 39.7%  | 40.5%              | 33.8%                 | 43.9%                                   | 35.9%          | **42.6%**     | 42.2%              | **51.4%**         | 47.6%        | 40.3%               | 35.1%           | 32.3%           | 39.6%         | 56.7%           | 48.0%             |

### How You Can Participate:
- **Use our benchmark**: Feel free to test your models using our benchmark and share your results.  
- **Submit checkpoints**: Alternatively, you can provide your model checkpoints, and we will evaluate them and update the leaderboard for you.

We look forward to your participation and contributions! 🌟

## News:
[2024/12/29] 🔥 The close-ended Leaderboard has been updated!

[2024/12/17] 🔥 The close-ended benchmark has been updated! [[Close-Ended Evaluation](https://github.com/hshjerry/VideoEspresso/tree/main/eval)]

[2024/12/16] 🔥 The test set has been released! Please check our huggingface repo. [[Test Set](https://huggingface.co/datasets/hshjerry0315/VideoEspresso-Test)]

## Overall View:
<p align="center" width="80%">
<img src="https://i.postimg.cc/LXzVcgFP/Wechat-IMG197.jpg"  width="100%" height="100%">
</p>

**Contact Us** 📧  
If you have any questions or want to submit your checkpoints, feel free to reach out to us via email:  

- [hshjerry0315@gmail.com](mailto:hshjerry0315@gmail.com)
- [aaron.weihuang@gmail.com](mailto:aaron.weihuang@gmail.com)

## Citation:
```
@article{han2024videoespresso,
  title={VideoEspresso: A Large-Scale Chain-of-Thought Dataset for Fine-Grained Video Reasoning via Core Frame Selection},
  author={Han, Songhao and Huang, Wei and Shi, Hairong and Zhuo, Le and Su, Xiu and Zhang, Shifeng and Zhou, Xu and Qi, Xiaojuan and Liao, Yue and Liu, Si},
  journal={arXiv preprint arXiv:2411.14794},
  year={2024}
}
```
