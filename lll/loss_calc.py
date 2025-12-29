import json
import matplotlib.pyplot as plt
import os  # 경로 확인을 위해 추가

def generate_loss_graph(input_path, output_path):
    epochs = []
    train_losses = []
    val_losses = []

    # [디버깅] 현재 어디서 실행 중인지, 파일은 있는지 확인
    print(f"📍 현재 작업 경로(CWD): {os.getcwd()}")
    print(f"🔎 찾으려는 파일 경로: {input_path}")

    if not os.path.exists(input_path):
        print("\n❌ 에러: 파일을 찾을 수 없습니다!")
        print("   -> 경로 오타가 있거나, 현재 실행 위치가 다를 수 있습니다.")
        print("   -> 절대 경로(예: /home/user/project/...)를 사용하는 것을 추천합니다.")
        return

    # [디버깅] 저장할 폴더가 없으면 만들기 (안 그러면 저장할 때 또 에러 남)
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"📂 저장 폴더가 없어서 생성했습니다: {output_dir}")

    try:
        with open(input_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                epochs.append(data['epoch'])
                train_losses.append(data['train_loss'])
                val_losses.append(data['val_loss'])

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, label='Training Loss', color='blue', linewidth=2)
        plt.plot(epochs, val_losses, label='Validation Loss', color='orange', linewidth=2)

        plt.title('Training and Validation Loss', fontsize=16)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5)

        plt.savefig(output_path)
        print(f"\n✅ 그래프 저장 완료: {output_path}")

    except Exception as e:
        print(f"❗ 실행 중 오류 발생: {e}")

# --- 설정 ---
if __name__ == "__main__":
    # ★★★ 여기에 '절대 경로'를 넣는 게 가장 확실합니다 ★★★
    # 예: "/home/deep/project/models_faster_..."
    input_file = 'models_faster_RCNN_Scratch_test/ood_50per_cropped/log.jsonl' 
    output_image = 'loss/50per_cropped.png'
    
    generate_loss_graph(input_file, output_image)