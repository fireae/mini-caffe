#ifndef CAFFE_CTC_DECODER_LAYER_HPP_
#define CAFFE_CTC_DECODER_LAYER_HPP_

#include <vector>

#include "../layer.hpp"


namespace caffe {

/**
 * @brief A layer that implements the decoder for a ctc
 *
 * Bottom blob is the probability of label and the sequence indicators.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */

class CTCDecoderLayer : public Layer {
 public:
  typedef vector<int> Sequence;
  typedef vector<Sequence> Sequences;

 public:
  explicit CTCDecoderLayer(const LayerParameter& param);
  virtual void LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top);

  virtual inline const char* type() const { return "CTCDecoder"; }

  // probabilities (T x N x C),
  // sequence_indicators (T x N),[optional]
  // target_sequences (T X N) [optional]
  // if a target_sequence is provided, an additional accuracy top blob is
  // required
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 3; }

  // sequences (terminated with negative numbers),
  // output scores [optional if 2 top blobs and bottom blobs = 2]
  // accuracy [optional, if target_sequences as bottom blob = 3]
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 3; }

  const Sequences& OutputSequences() const {return output_sequences_;}

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top);

  virtual void Backward_cpu(const vector<Blob*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob*>& bottom);

  virtual void Decode(const Blob* probabilities,
                      const Blob* sequence_indicators,
                      Sequences* output_sequences,
                      Blob* scores) const = 0;

  virtual void Decode(const Blob* probabilities,
	  Sequences* output_sequences,
	  Blob* scores) const = 0;

  int EditDistance(const Sequence &s1, const Sequence &s2);

 protected:
  Sequences output_sequences_;
  int T_;
  int N_;
  int C_;
  int blank_index_;
  bool merge_repeated_;

  int sequence_index_;
  int score_index_;
  int accuracy_index_;
};


class CTCGreedyDecoderLayer : public CTCDecoderLayer {
 private:
  using typename CTCDecoderLayer::Sequences;
  using CTCDecoderLayer::T_;
  using CTCDecoderLayer::N_;
  using CTCDecoderLayer::C_;
  using CTCDecoderLayer::blank_index_;
  using CTCDecoderLayer::merge_repeated_;

 public:
  explicit CTCGreedyDecoderLayer(const LayerParameter& param)
      : CTCDecoderLayer(param) {}

  virtual inline const char* type() const { return "CTCGreedyDecoder"; }

 protected:
  virtual void Decode(const Blob* probabilities,
                      const Blob* sequence_indicators,
                      Sequences* output_sequences,
                      Blob* scores) const;

  virtual void Decode(const Blob* probabilities,
	  Sequences* output_sequences,
	  Blob* scores) const;

};

}  // namespace caffe

#endif  // CAFFE_CTC_DECODER_LAYER_HPP_
