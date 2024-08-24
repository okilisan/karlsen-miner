use crate::proto::{
    karlsend_message::Payload, GetBlockTemplateRequestMessage, GetInfoRequestMessage, KarlsendMessage,
    NotifyBlockAddedRequestMessage, NotifyNewBlockTemplateRequestMessage, RpcBlock, SubmitBlockRequestMessage,
};
use crate::{
    pow::{self, HeaderHasher},
    Hash,
};

impl KarlsendMessage {
    #[inline(always)]
    pub fn get_info_request() -> Self {
        KarlsendMessage { payload: Some(Payload::GetInfoRequest(GetInfoRequestMessage {})) }
    }
    #[inline(always)]
    pub fn notify_block_added() -> Self {
        KarlsendMessage { payload: Some(Payload::NotifyBlockAddedRequest(NotifyBlockAddedRequestMessage {})) }
    }

    #[inline(always)]
    pub fn submit_block(block: RpcBlock) -> Self {
        KarlsendMessage {
            payload: Some(Payload::SubmitBlockRequest(SubmitBlockRequestMessage {
                block: Some(block),
                allow_non_daa_blocks: false,
            })),
        }
    }
}

impl From<GetInfoRequestMessage> for KarlsendMessage {
    fn from(a: GetInfoRequestMessage) -> Self {
        KarlsendMessage { payload: Some(Payload::GetInfoRequest(a)) }
    }
}
impl From<NotifyBlockAddedRequestMessage> for KarlsendMessage {
    fn from(a: NotifyBlockAddedRequestMessage) -> Self {
        KarlsendMessage { payload: Some(Payload::NotifyBlockAddedRequest(a)) }
    }
}

impl From<GetBlockTemplateRequestMessage> for KarlsendMessage {
    fn from(a: GetBlockTemplateRequestMessage) -> Self {
        KarlsendMessage { payload: Some(Payload::GetBlockTemplateRequest(a)) }
    }
}

impl From<NotifyNewBlockTemplateRequestMessage> for KarlsendMessage {
    fn from(a: NotifyNewBlockTemplateRequestMessage) -> Self {
        KarlsendMessage { payload: Some(Payload::NotifyNewBlockTemplateRequest(a)) }
    }
}

impl RpcBlock {
    #[inline(always)]
    pub fn block_hash(&self) -> Option<Hash> {
        let mut hasher = HeaderHasher::new();
        pow::serialize_header(&mut hasher, self.header.as_ref()?, false);
        Some(hasher.finalize())
    }
}
