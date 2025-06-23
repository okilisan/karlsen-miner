use log::info;

use crate::proto::SubmitPouwResultRequestMessage;
use crate::proto::{
    karlsend_message::Payload, GetBlockTemplateRequestMessage, GetInfoRequestMessage, GetPouwTaskRequestMessage,
    KarlsendMessage, NotifyBlockAddedRequestMessage, NotifyNewBlockTemplateRequestMessage, RpcBlock,
    SubmitBlockRequestMessage,
};
use crate::{
    pow::{self, HeaderHasher},
    Hash,
};

impl KarlsendMessage {
    #[must_use]
    #[inline(always)]
    pub fn get_info_request() -> Self {
        KarlsendMessage { payload: Some(Payload::GetInfoRequest(GetInfoRequestMessage {})) }
    }
    #[must_use]
    #[inline(always)]
    pub fn notify_block_added() -> Self {
        KarlsendMessage { payload: Some(Payload::NotifyBlockAddedRequest(NotifyBlockAddedRequestMessage {})) }
    }

    #[must_use]
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
    #[inline(always)]
    fn from(a: GetInfoRequestMessage) -> Self {
        KarlsendMessage { payload: Some(Payload::GetInfoRequest(a)) }
    }
}
impl From<NotifyBlockAddedRequestMessage> for KarlsendMessage {
    #[inline(always)]
    fn from(a: NotifyBlockAddedRequestMessage) -> Self {
        KarlsendMessage { payload: Some(Payload::NotifyBlockAddedRequest(a)) }
    }
}

impl From<GetBlockTemplateRequestMessage> for KarlsendMessage {
    #[inline(always)]
    fn from(a: GetBlockTemplateRequestMessage) -> Self {
        KarlsendMessage { payload: Some(Payload::GetBlockTemplateRequest(a)) }
    }
}

impl From<NotifyNewBlockTemplateRequestMessage> for KarlsendMessage {
    #[inline(always)]
    fn from(a: NotifyNewBlockTemplateRequestMessage) -> Self {
        KarlsendMessage { payload: Some(Payload::NotifyNewBlockTemplateRequest(a)) }
    }
}

impl From<GetPouwTaskRequestMessage> for KarlsendMessage {
    #[inline(always)]
    fn from(a: GetPouwTaskRequestMessage) -> Self {
        info!("GetPouwTaskRequestMessage payload: {:}", a.subnet);
        KarlsendMessage { payload: Some(Payload::GetPouwTaskRequest(a)) }
    }
}

impl From<SubmitPouwResultRequestMessage> for KarlsendMessage {
    #[inline(always)]
    fn from(a: SubmitPouwResultRequestMessage) -> Self {
        info!("SubmitPouwResultRequestMessage task_id: {:} - data: {:}", a.task_id, a.data);
        KarlsendMessage { payload: Some(Payload::SubmitPouwResultRequest(a)) }
    }
}

impl RpcBlock {
    #[must_use]
    #[inline(always)]
    pub fn block_hash(&self) -> Option<Hash> {
        let mut hasher = HeaderHasher::new();
        pow::serialize_header(&mut hasher, self.header.as_ref()?, false);
        Some(hasher.finalize())
    }
}
