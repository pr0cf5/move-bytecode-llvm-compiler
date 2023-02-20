// Copyright (c) Aptos
// SPDX-License-Identifier: Apache-2.0

use futures::channel::{mpsc::SendError, oneshot::Canceled};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Debug, Deserialize, Error, PartialEq, Eq, Serialize)]
pub enum Error {
    #[error("The requested data is unavailable and cannot be found in the network! Error: {0}")]
    DataIsUnavailable(String),
    #[error("Error returned by the aptos data client: {0}")]
    AptosDataClientError(String),
    #[error("The response from the aptos data client is invalid! Error: {0}")]
    AptosDataClientResponseIsInvalid(String),
    #[error("An integer overflow has occurred: {0}")]
    IntegerOverflow(String),
    #[error("No data to fetch: {0}")]
    NoDataToFetch(String),
    #[error("Unexpected error encountered: {0}")]
    UnexpectedErrorEncountered(String),
    #[error("Encountered an unsupported request: {0}")]
    UnsupportedRequestEncountered(String),
}

impl Error {
    /// Returns a summary label for the error
    pub fn get_label(&self) -> &'static str {
        match self {
            Self::DataIsUnavailable(_) => "data_is_unavailable",
            Self::AptosDataClientError(_) => "aptos_data_client_error",
            Self::AptosDataClientResponseIsInvalid(_) => "aptos_data_client_response_is_invalid",
            Self::IntegerOverflow(_) => "integer_overflow",
            Self::NoDataToFetch(_) => "no_data_to_fetch",
            Self::UnexpectedErrorEncountered(_) => "unexpected_error_encountered",
            Self::UnsupportedRequestEncountered(_) => "unsupported_request_encountered",
        }
    }
}

impl From<aptos_data_client::Error> for Error {
    fn from(error: aptos_data_client::Error) -> Self {
        Error::AptosDataClientError(error.to_string())
    }
}

impl From<SendError> for Error {
    fn from(error: SendError) -> Self {
        Error::UnexpectedErrorEncountered(error.to_string())
    }
}

impl From<Canceled> for Error {
    fn from(canceled: Canceled) -> Self {
        Error::UnexpectedErrorEncountered(canceled.to_string())
    }
}
